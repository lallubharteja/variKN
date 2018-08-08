// The main library for writing out the probabilities
#include <limits>
#include "GramSorter.hh"
#ifdef _WIN32
inline double log2( double n ){
	return log(n) / log(2.0);
}
#endif

/************************************************************
 Non-inlined functions for the different child classes      *
*************************************************************/

template <typename KT, typename ICT>
ProbsLM_impl<KT, ICT>::ProbsLM_impl(
  const std::string data, const std::string vocab,
  const int order, Storage_t<KT, ICT> *datastorage, 
  const std::string sent_boundary, const std::string read_prob, const bool average,
  const indextype hashsize):
  ProbsLM_t<KT, ICT>(data)
{
  this->mop = new MultiOrderProbs<KT, ICT>;
  this->read_prob = read_prob;
  this->average = average;
  this->constructor_helper(vocab, order, datastorage, 
		     hashsize, sent_boundary);
}

template <typename KT, typename ICT> void ProbsLM_impl<KT, ICT>::
remove_sent_start_prob_fbase(ICT *dummy) {
  std::vector<KT> tmp(1, (KT) this->m_sent_boundary);
  const ICT value=this->mop->GetCount(tmp);
  this->mop->IncrementCount(tmp,-value);
}

template <typename KT, typename CT>
void ProbsLM_t<KT, CT>::add_zeroprob_grams_fbase(CT *dummy) {
  std::vector<KT> v;
  CT num;
  if (std::numeric_limits<KT>::max() < this->vocab.num_words()) {
    fprintf(stderr, "Too large vocab for the given key type!!! Abort.\n");
    exit(-1);
  }


  this->set_order(mop->order());

  for (int o=this->m_order;o>=2;o--) {
    v.resize(o);
    this->mop->StepCountsOrder(true,o,&v[0],&num);
    while (this->mop->StepCountsOrder(false,o,&v[0],&num)) {
      this->mop->IncrementCount(o-1,&v[0], (CT) 0);
    }
    CT b;
    
    if (o==2) continue;
    this->mop->StepBackoffsOrder(true, o, &v[0], &b);
    while (this->mop->StepBackoffsOrder(false, o, &v[0], &b))
      if (b>0) this->mop->IncrementCount(o-1, &v[0], (CT) 0);
  }

  // Add zeroprob unigrams:
  for (KT i=0;i<this->vocab.num_words();i++) {
    //fprintf(stderr, "%d/%d %s\n", i, this->vocab.num_words(), this->vocab.word(i).c_str());
    this->mop->IncrementCount(1,&i, (CT) 0);
  }
}

#ifdef sgi
#define log2 1/flog10(2)*flog10
#endif

template <typename KT, typename CT>
void ProbsLM_t<KT, CT>::constructor_helper(
  const std::string &vocabname, const int order, Storage_t<KT, CT> *datastorage,
  const indextype hashsize, 
  const std::string &sent_start_symbol){

  mop->hashsize=hashsize;
  mop->average = this->average;
  io::Stream::verbose=true;
  this->input_data_size=0;

  if (vocabname.length()) {
    fprintf(stderr,"Using vocab %s\n", vocabname.c_str());
    if (this->vocab.num_words()>1) 
      fprintf(stderr,"Warning: something is going wrong. The vocabularies must be the same (not checked)\n");
    if (this->vocab.num_words()>65534) { 
      fprintf(stderr,"Too big vocabulary for --smallvocab (%d). Exit.\n", 
		this->vocab.num_words());
	exit(-1);
    }
    io::Stream vocabin(vocabname,"r");
    this->vocab.read(vocabin.file);
    if (this->vocab.num_words()<1) {
      fprintf(stderr, "Warning: no words from vocab file? Exit\n");
      exit(-1);
    }
  }
  if (this->vocab.num_words()>1) {
    fprintf(stderr,"Restricted vocab\n");
    //fprintf(stderr,"Init from text\n");
    if (!datastorage) {
      //fprintf(stderr,"Init from text\n");
      io::Stream datain(this->m_data_name,"r");
      this->input_data_size=mop->InitializeCountsFromText(datain.file, &(this->vocab), false, order, sent_start_symbol);
      datain.close();
    } else {
      //fprintf(stderr,"Init from file\n");
      io::Stream probin(this->read_prob,"r");
      datastorage->read_prob(probin.file, this->vocab);
      //fprintf(stderr,"All read\n");
      int sent_start_idx=this->vocab.word_index(sent_start_symbol);
      if (sent_start_idx==0 && sent_start_symbol.length()) {
        fprintf(stderr,"No sentence start %s(len %d) in vocab, exit.\n", sent_start_symbol.c_str(), (int) sent_start_symbol.length());
        exit(-1);
      }
      this->input_data_size=mop->InitializeCountsFromStorage(datastorage, order, sent_start_idx);
      probin.close();
    }
    //this->init_probs(order, sent_start_symbol);
  } else {
    if ( this->KT_is_short((KT *)NULL) ) {
      NgramCounts_t<int, CT> nc_tmp(1,0,500000);
      io::Stream datain(this->m_data_name,"r");
      nc_tmp.count(datain.file, true);
      datain.close();
      nc_tmp.vocab->copy_vocab_to(this->vocab);
      datain.open(this->m_data_name,"r");
      this->input_data_size=mop->InitializeCountsFromText(datain.file, &(this->vocab), false, order, sent_start_symbol);
      datain.close();
      //this->init_probs(order, sent_start_symbol);
     } else {
      io::Stream datain(this->m_data_name,"r"); 
      this->input_data_size=mop->InitializeCountsFromText(datain.file, &(this->vocab), true, order, sent_start_symbol);
      datain.close();
      if (datastorage) {
	io::Stream probin(this->read_prob,"r");
	datastorage->read_prob(probin.file, this->vocab);
        probin.close();
      }     
      //this->init_probs(order, sent_start_symbol);
     }
  }

  this->set_order(mop->order());
  if (sent_start_symbol.size()) this->m_sent_boundary=this->vocab.word_index(sent_start_symbol);

  fprintf(stderr,"Estimating bo counts\n");
  estimate_bo_counts();
  
  //sentence boundary counts are important for some ngrams and hence, not removed
  //remove_sent_start_prob();

  fprintf(stderr,"Reading probabilities\n");
  this->init_probs(order, datastorage, sent_start_symbol);

  this->model_cost_scale=log2(this->vocab.num_words())+2*10;
}

template <typename KT, typename ICT>
void ProbsLM_t<KT, ICT>::init_probs(const int order, Storage_t<KT, ICT> *datastorage, const std::string &sent_start_symbol ) {
  int sent_start_idx;
  if (datastorage) {
    sent_start_idx=this->vocab.word_index(sent_start_symbol);
    if (sent_start_idx==0 && sent_start_symbol.length()) {
      fprintf(stderr,"No sentence start %s(len %d) in vocab, exit.\n", sent_start_symbol.c_str(), (int) sent_start_symbol.length());
      exit(-1);
    }
  }
  for (int o=1; o<=order; o++) {
    io::Stream probin(this->read_prob, "r");
    if (!datastorage)
      mop->InitializeProbsFromText(probin.file, &(this->vocab), false, o, sent_start_symbol);
    else 
      mop->InitializeProbsFromStorage(datastorage, order, sent_start_idx);
    probin.close();
  }
}

template <typename KT, typename CT>
ProbsLM_t<KT, CT>::~ProbsLM_t() {
  delete mop;
}

template <typename KT, typename ICT>
void ProbsLM_impl<KT, ICT>::set_order(int o) {
  this->m_order=o;  
}

template <typename KT, typename CT>
double ProbsLM_t<KT, CT>::logprob_file(const char *name) {
  double logprob=0.0;

  std::vector<KT> indices;
  indices.reserve(this->m_order);
  io::Stream f(name,"r");
  char w[MAX_WLEN+1];
  long nwords=0;
  while (fscanf(f.file,MAX_WLEN_FMT_STRING,w)==1) {
    const KT idx=this->m_ng->word_index(w);
    if (idx==this->m_sent_boundary) {
      indices.clear();
      indices.push_back(idx);
      continue;
    } else if (indices.size()<this->m_order) indices.push_back(idx);
    else { 
      for (int i=0;i<this->m_order-1;i++) indices[i]=indices[i+1];
      indices[this->m_order-1]=idx;
      //fprintf(stderr,"read %s(%d)\n",w,indices[m_order-1]);
    }
    if (indices.back()==0) continue; // Do not optimize for unks...
    nwords++;
    //fprintf(stderr,"%s %.1g\n",w,tableprob(indices));
    logprob+=safelogprob(tableprob(indices));
  }
  f.close();
  return(-logprob/nwords);
}

template <typename KT, typename CT>
void ProbsLM_t<KT, CT>::estimate_bo_counts() {
  /* Estimate backoff and lower order counts            */
  std::vector<KT> v(this->m_order);
  CT value, prob;

  if (this->m_sent_boundary<0) {
    for (int o=this->m_order;o>=1;o--) {
      mop->StepCountsOrder(true,o,&v[0],&value);
      while (mop->StepCountsOrder(false,o,&v[0],&value)) {
	if (o>1) mop->IncrementCount(o-1,&v[1],value);
        else mop->m_uni_counts_den += value;
      }
    }
    return;
  } 
  
  for (int o=this->m_order;o>=1;o--) {
    mop->StepCountsOrder(true,o,&v[0],&value);
    while (mop->StepCountsOrder(false,o,&v[0],&value)) {
      if (o>1) mop->IncrementCount(o-1,&v[1],value);
      else mop->m_uni_counts_den += value;
    }
  }
}

template <typename KT, typename ICT>
ICT ProbsLM_impl<KT, ICT>::text_prob(const int order, const KT *i)
{
  return this->mop->GetProb(order,i);
}

template <typename KT, typename ICT> ICT ProbsLM_impl<KT, ICT>::
text_coeff(const int order, const KT *i){
  if (order>this->m_order) return(1.0);

  ICT bb;
  this->mop->GetBackoff(order,i,&bb);
  //fprintf(stderr," BO (%.5f * %.d +%d)/ %d\n", m_discount[order], bb.nzer, bb.prune_den ,bb.den);
  if (bb) return ((ICT) 1.0) - bb ;
  return (ICT) 1.0;
}

template <typename KT, typename CT>
void ProbsLM_t<KT, CT>::remove_zeroprob_grams() {
  for (int o=mop->m_counts.size()-1;o>=2;o--) {
    sikMatrix<KT, CT> *m=mop->m_counts[o];
    for (indextype i=0;i<m->num_entries();i++) 
      if (*(m->Idx2Valp(i))<1e-3)
	RemoveEntryIdx(m->m,i--);
  }
}

template <typename KT, typename CT> void ProbsLM_t<KT, CT>::
create_model(float prunetreshold) {
  //fprintf(stderr,"Worlds are colliding\n");
  if (prunetreshold>0.0 || this->discard_ngrams_with_unk) {
    prune_model(prunetreshold, (Storage_t<KT, CT> *) NULL);
  } 
}
template <typename KT, typename CT>
void ProbsLM_t<KT, CT>::probs2ascii(FILE *out) {
  mop->WriteProbs(out);
}

template <typename KT, typename CT>
void ProbsLM_t<KT, CT>::counts2lm(FILE *out) {
  /*********************************************/
  /* Put smoothed model into ngram            */
  /*******************************************/

  TreeGram::Gram gr;
  float prob,coeff;
  std::vector<KT> v(1);
  CT num;
  std::string field_separator=" ";

  // Header containing counts for each order
  fprintf(out, "\\data\\\n");
  for (int i = 1; i <= this->m_order; i++) {
    if (mop->order_size(i)) fprintf(out, "ngram %d=%d\n", i, mop->order_size(i));
  }

  for (int o=1;o<=this->m_order;o++) {
    //fprintf(stderr,"Adding o %d\n",o);
    bool breaker=true;
    v.resize(o);
    gr.resize(o);
    GramSorter gramsorter(o,mop->order_size(o));
    mop->StepCountsOrder(true,o,&v[0],&num);
    while (mop->StepCountsOrder(false,o,&v[0],&num)) {
      prob=text_prob(o,&v[0]);
      coeff=text_coeff(o+1,&v[0]);
      for (int i=0;i<o;i++) gr[i]=v[i];
      //fprintf(stderr,"to sorter: %.4f ",safelogprob(prob));print_indices(stderr, v); fprintf(stderr," %.4f\n", safelogprob(coeff));
      gramsorter.add_gram(gr,safelogprob(prob),safelogprob2(coeff));
      breaker=false;
    }
    if (breaker) break;
    gramsorter.sort();

    fprintf(out, "\n\\%d-grams:\n",o);
    
    for (size_t i = 0; i < gramsorter.num_grams(); i++) {
      GramSorter::Data data = gramsorter.data(i);
      gr = gramsorter.gram(i);
      
      fprintf(out, "%g", data.log_prob);
      fprintf(out, "%s%s", field_separator.c_str(),
              (this->vocab.word(gr[0])).c_str());
      for (int j = 1; j < o; j++) {
        fprintf(out, " %s", (this->vocab.word(gr[j])).c_str());
      }
      //fprintf(stderr,"adding ");print_indices(stderr, gr);fprintf(stderr," %.4f %.4f\n", data.log_prob, data.back_off);
      
      if (data.back_off != 0.0)
        fprintf(out, "%s%g\n", field_separator.c_str(), data.back_off);
      else
        fprintf(out, "\n");
    } 
  }
  fprintf(out, "\n\\end\\\n");
}

template <typename KT, typename CT>
CT ProbsLM_t<KT, CT>::tableprob(std::vector<KT> &indices) {
  CT prob= (CT) 0;
  KT *iptr;

  //fprintf(stderr,"looking ");print_indices(stderr,indices);fprintf(stderr,"\n");
  const int looptill=std::min(indices.size(),(size_t) this->m_order);
  for (int n=1;n<=looptill;n++) {
    iptr=&(indices.back())-n+1;
    if (n>1) {
      //fprintf(stderr,"prob %.4f * coeff %.4f = ", prob, text_coeff(n, iptr));
      prob*=text_coeff(n,iptr);
      //fprintf(stderr,"%.4f\n", prob);
    }
    //fprintf(stderr,"oldprob %.4f + prob %.4f = ", prob, text_prob(n, iptr));
    prob += text_prob(n,iptr);
    //fprintf(stderr,"%.4f\n",prob);
  }
  //fprintf(stderr,"vc: return %e\n",prob);
  assert(prob>=-1e-03 && prob<=1.001);
  return(prob);
}

template <typename KT, typename CT>
void ProbsLM_t<KT, CT>::prune_model_fbase
(float threshold, Storage_t<KT, CT> *real_counts) {
  std::vector<KT> v;
  CT num;
  float logprobdelta, safelogprob_mult;

  threshold=threshold*this->model_cost_scale;
  this->set_order(mop->order());
  for (int o=this->order();o>=2;o--) {
    if (real_counts) {
      fprintf(stderr,"Using real counts\n");
      real_counts->initialize_fast_search_lists_for_pruning(o, mop->m_counts[o]);
    }

    fprintf(stderr,"Pruning order %d\n", o);
    v.resize(o);
    mop->StepCountsOrder(true,o,&v[0],&num);
    while (mop->StepCountsOrder(false,0,&v[0],&num)) {
      if (num==0) {
	if (!real_counts) mop->DeleteCurrentST(o);
	continue;
      }
      
      assert(num>0);
      mop->ResetCaches();
      
      if (this->discard_ngrams_with_unk) {
        bool flag=false;
        for (int ngu=0;ngu<o;ngu++) {
          if (v[ngu]==0) {
            flag=true;
            break;
          }
        }
        if (flag) {
          prune_gram(v, num);
	  if (!real_counts) mop->DeleteCurrentST(o);
          continue;
        }
      }
      
      if (!(num>0)) fprintf(stderr,"Weird num %ld\n", (long) num);
      //print_indices(stderr,v); fprintf(stderr,"IDX %d\n", idx);
      if (real_counts) {
 	const indextype idx = FindEntry(mop->m_counts[o]->m, (byte *) (&v[0]), 0);
	//print_indices(stderr,v); fprintf(stderr,"IDX %d\n", idx);
	safelogprob_mult=real_counts->prune_lists[idx];
      } else {
	safelogprob_mult=num;
      }

      logprobdelta=safelogprob(tableprob(v));

      /* Try removing the gram */
      prune_gram(v, num);

      logprobdelta-=safelogprob(tableprob(v));
      logprobdelta*=safelogprob_mult;

      if (logprobdelta-threshold>0) {
        mop->UndoCached();
      } else if (!real_counts) {
        mop->DeleteCurrentST(o);
      }
    }
    // Clean up for realcounts
    if (real_counts) {
      mop->StepCountsOrder(true,o,&v[0],&num);
      while (mop->StepCountsOrder(false,0,&v[0],&num)) {
	if (num==0) mop->DeleteCurrentST(o);
      }
    }
  }
  // Clean backoffs 
  mop->RemoveDefaultBackoffs();
  if (real_counts) real_counts->prune_lists.clear();
}

template <typename KT, typename ICT>
void ProbsLM_impl<KT, ICT>::prune_gram(
  std::vector<KT> &v, ICT num) {
  ICT prob;
  const int o=v.size();

  this->mop->IncrementCountCache(o,&v[0],-num);
  prob=this->mop->GetProb(v);
  this->mop->IncrementBackoffCache(o,&v[0],-prob);
  this->mop->IncrementProbCache(o,&v[0],-prob);

}
