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
ProbsAbs_int_disc<KT, ICT>::ProbsAbs_int_disc(
  const std::string data, const std::string vocab,
  const int order,
  const std::string sent_boundary, const std::string read_prob, const bool average,
  const indextype hashsize):
  ProbsAbs_t<KT, ICT>(data)
{
  this->moc = new MultiOrderCounts_1nzer<KT, ICT>;
  this->read_prob = read_prob;
  this->average = average;
  this->constructor_helper(vocab, order, 
		     hashsize, sent_boundary);
}

template <typename KT, typename ICT>
void ProbsAbs_int_disc<KT, ICT>::estimate_nzer_counts() {
  std::vector<KT> v(this->m_order);
  ICT value;

  for (int o=1;o<=this->m_order;o++) {
    this->moc->StepCountsOrder(true, o, &v[0], &value);
    while (this->moc->StepCountsOrder(false, o, &v[0], &value)) {
      if (value==0) continue;
      this->moc->IncrementBackoffNzer(o,&v[0], (ICT) 1); 
    }
  }
}

template <typename KT, typename ICT> template <typename BOT> void ProbsAbs_int_disc<KT, ICT>::
remove_sent_start_prob_fbase(BOT *dummy) {
  std::vector<KT> tmp(1, (KT) this->m_sent_boundary);
  const ICT value=this->moc->GetCount(tmp);
  this->moc->IncrementCount(tmp,-value);
  const ICT prob=this->moc->GetProb(tmp);
  this->moc->IncrementProb(tmp,-prob);
  BOT bo;
  this->moc->zero_bo(bo);
  bo.den=-prob;bo.nzer=-1.0;
  this->moc->IncrementBackoff(1,NULL,&bo);
}

template <typename KT, typename CT> template <typename BOT> 
void ProbsAbs_t<KT, CT>::add_zeroprob_grams_fbase(BOT *dummy) {
  std::vector<KT> v;
  CT num;
  if (std::numeric_limits<KT>::max() < this->vocab.num_words()) {
    fprintf(stderr, "Too large vocab for the given key type!!! Abort.\n");
    exit(-1);
  }


  this->set_order(moc->order());

  for (int o=this->m_order;o>=2;o--) {
    v.resize(o);
    this->moc->StepCountsOrder(true,o,&v[0],&num);
    while (this->moc->StepCountsOrder(false,o,&v[0],&num)) {
      this->moc->IncrementCount(o-1,&v[0], (CT) 0);
    }
    BOT b;
    
    if (o==2) continue;
    this->moc->StepBackoffsOrder(true, o, &v[0], &b);
    while (this->moc->StepBackoffsOrder(false, o, &v[0], &b))
      if (b.den>0) this->moc->IncrementCount(o-1, &v[0], (CT) 0);
  }

  // Add zeroprob unigrams:
  for (KT i=0;i<this->vocab.num_words();i++) {
    //fprintf(stderr, "%d/%d %s\n", i, this->vocab.num_words(), this->vocab.word(i).c_str());
    this->moc->IncrementCount(1,&i, (CT) 0);
  }
}

#ifdef sgi
#define log2 1/flog10(2)*flog10
#endif

template <typename KT, typename CT>
void ProbsAbs_t<KT, CT>::constructor_helper(
  const std::string &vocabname, const int order,  
  const indextype hashsize, 
  const std::string &sent_start_symbol){

  moc->hashsize=hashsize;
  moc->average = this->average;
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
    io::Stream datain(this->m_data_name,"r");
    this->input_data_size=moc->InitializeCountsFromText(datain.file, &(this->vocab), false, order, sent_start_symbol);
    datain.close();
    //this->init_probs(order, sent_start_symbol);
  } else {
    if ( this->KT_is_short((KT *)NULL) ) {
      NgramCounts_t<int, CT> nc_tmp(1,0,500000);
      io::Stream datain(this->m_data_name,"r");
      nc_tmp.count(datain.file, true);
      datain.close();
      nc_tmp.vocab->copy_vocab_to(this->vocab);
      datain.open(this->m_data_name,"r");
      this->input_data_size=moc->InitializeCountsFromText(datain.file, &(this->vocab), false, order, sent_start_symbol);
      datain.close();
      //this->init_probs(order, sent_start_symbol);
     } else {
      io::Stream datain(this->m_data_name,"r"); 
      this->input_data_size=moc->InitializeCountsFromText(datain.file, &(this->vocab), true, order, sent_start_symbol);
      datain.close();
      //this->init_probs(order, sent_start_symbol);
     }
  }

  this->set_order(moc->order());
  if (sent_start_symbol.size()) this->m_sent_boundary=this->vocab.word_index(sent_start_symbol);

  fprintf(stderr,"Estimating bo counts\n");
  estimate_bo_counts();

  fprintf(stderr,"Reading probabilities\n");
  this->init_probs(order, sent_start_symbol);
  //remove_sent_start_prob();

  fprintf(stderr,"Estimating nzer counts\n");
  this->estimate_nzer_counts();


  //this->model_cost_scale=log2(this->vocab.num_words())+2*10;
}

template <typename KT, typename ICT>
void ProbsAbs_t<KT, ICT>::init_probs(const int order, const std::string &sent_start_symbol ) {
  for (int o=1; o<=order; o++) {
    io::Stream probin(this->read_prob, "r");
    moc->InitializeProbsFromText(probin.file, &(this->vocab), false, o, sent_start_symbol);
    probin.close();
  }
}

template <typename KT, typename CT>
ProbsAbs_t<KT, CT>::~ProbsAbs_t() {
  delete moc;
}

template <typename KT, typename ICT>
void ProbsAbs_int_disc<KT, ICT>::set_order(int o) {
  this->m_order=o;  
}

template <typename KT, typename CT>
double ProbsAbs_t<KT, CT>::logprob_file(const char *name) {
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
void ProbsAbs_t<KT, CT>::estimate_bo_counts() {
  /* Estimate backoff and lower order counts            */
  std::vector<KT> v(this->m_order);
  CT value, prob;

  if (this->m_sent_boundary<0) {
    for (int o=this->m_order;o>=1;o--) {
      moc->StepCountsOrder(true,o,&v[0],&value);
      while (moc->StepCountsOrder(false,o,&v[0],&value)) {
        //prob=moc->GetProb(o,&v[0]);
	//moc->IncrementBackoffDen(o,&v[0],prob);
	if (o>1) moc->IncrementCount(o-1,&v[1],value);
        else moc->m_uni_counts_den += value;
      }
    }
    return;
  } 
  
  // Ugly way of doing things...
  for (int o=this->m_order;o>=1;o--) {
    moc->StepCountsOrder(true,o,&v[0],&value);
    while (moc->StepCountsOrder(false,o,&v[0],&value)) {
      //prob=moc->GetProb(o,&v[0]);
      /*bool flag=false;
      for (int i=1;i<o;i++) {
	if (v[i]==this->m_sent_boundary) {
	  moc->DeleteCurrentST(o);
	  flag=true;
	  break;
	}
      }*/
      if (o>1) moc->IncrementCount(o-1,&v[1],value);
      else moc->m_uni_counts_den += value;
      //if (!flag) moc->IncrementBackoffDen(o,&v[0],prob);
    }
  }
}

template <typename KT, typename CT>
void ProbsAbs_t<KT, CT>::remove_zeroprob_grams() {
  for (int o=moc->m_counts.size()-1;o>=2;o--) {
    sikMatrix<KT, CT> *m=moc->m_counts[o];
    for (indextype i=0;i<m->num_entries();i++) 
      if (*(m->Idx2Valp(i))<1e-3)
	RemoveEntryIdx(m->m,i--);
  }
}

template <typename KT, typename CT>
void ProbsAbs_t<KT, CT>::probs2ascii(FILE* out) {
  moc->WriteProbs(out);
}
