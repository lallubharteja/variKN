// Library for storing and modifying the probabilities of ngrams

template <typename KT, typename CT>
float MultiOrderProbs<KT, CT>::GetProb(const std::vector<KT> &v) {
  return(GetProb(v.size(),&v[0]));
}

template <typename KT, typename CT>
CT MultiOrderProbs<KT, CT>::GetCount(const std::vector<KT> &v) {
  return(GetCount(v.size(),&v[0]));
}

template <typename KT, typename CT>
CT MultiOrderProbs<KT,CT>::IncrementCount(const std::vector<KT> &v, 
					 const CT value) {
  return(IncrementCount(v.size(), &v[0],value));
}

template <typename KT, typename CT>
void MultiOrderProbs<KT,CT>::IncrementProb(const std::vector<KT> &v, 
					 const CT value) {
  IncrementProb(v.size(), &v[0],value);
  return;
}

/***********************************************************************
  Functions for the lower level template implementing most of the class
**********************************************************************/

template <typename KT, typename CT>
long MultiOrderProbs<KT, CT>::InitializeCountsFromText(FILE *in, Vocabulary *vocab, const bool grow_vocab, const int read_order, const std::string &sent_start_sym) {
  char charbuf[MAX_WLEN+1];
  long num_read=0;
  int sent_start_idx;
  KT idx;
  std::vector<KT> v;
  
  if (grow_vocab) {
    if (sent_start_sym.size()) sent_start_idx=vocab->add_word(sent_start_sym);
    else sent_start_idx=-1;
    vocabsize=64000;
  } else {
    vocabsize=vocab->num_words();
    if (!sent_start_sym.size()) sent_start_idx=-1;
    else if (!(sent_start_idx=vocab->word_index(sent_start_sym))) {
      fprintf(stderr,"No sentence start symbol %s in vocabulary, exit.\n", sent_start_sym.c_str());
      exit(-1);
    }
  }

  while (fscanf(in,MAX_WLEN_FMT_STRING,charbuf)!=EOF) {
    num_read++;
    //if (num_read % 1000000 == 0) fprintf(stderr,"Read %lld words\n",num_read);
    if (grow_vocab) idx=vocab->add_word(charbuf);
    else idx=vocab->word_index(charbuf);

    //fprintf(stderr,"Word %s = %d\n", charbuf, idx);

    if (idx==sent_start_idx) v.clear();

    if (v.size()<read_order) v.push_back(idx);
    else v.back()=idx;

    IncrementCount(v, (CT) 1);
    //fprintf(stderr,"Cadd ");print_indices(stderr,v);fprintf(stderr," %f\n", (CT) 1);

    if (v.size()==read_order) 
      for (int j=0;j<read_order-1;j++) 
	v[j]=v[j+1];
  }
  fprintf(stderr,"Finished reading %ld words.\n", num_read);
  return(num_read);
}

template <typename KT, typename CT>
long MultiOrderProbs<KT, CT>::InitializeCountsFromStorage(Storage_t<KT, CT> *data, const int read_order, const int sent_start_idx) {
  long num_read=0;
  std::vector<KT> v;
  KT idx;
  //fprintf(stderr,"Init from storage\n");

  for (size_t di=0;di<data->size(); di++) {
    //fprintf(stderr,"adding %d: %d\n", di, data->data(di));
    num_read++;
    idx=data->data(di);
    if (idx==sent_start_idx) v.clear();

    if (v.size()<read_order) v.push_back(idx);
    else v.back()=idx;

    IncrementCount(v,1);

    if (v.size()==read_order) 
      for (int j=0;j<read_order-1;j++) 
	v[j]=v[j+1];
  }
  return(num_read);
}

/**
 * The function below assumes it will be called after the
 * counts from the text have been read in and populated.
**/
template <typename KT, typename CT>
long MultiOrderProbs<KT, CT>::InitializeProbsFromText(FILE *in, Vocabulary *vocab, const bool grow_vocab, const int read_order, const std::string &sent_start_sym) {
  char charbuf[MAX_WLEN+1];
  long num_read=0;
  int sent_start_idx;
  KT idx;
  std::vector<KT> v;
  TreeGram::Gram gr;
  CT prob= 0.0, den;
  //fprintf(stderr,"Order %i\n", read_order);

  if (grow_vocab) {
    if (sent_start_sym.size()) sent_start_idx=vocab->add_word(sent_start_sym);
    else sent_start_idx=-1;
    vocabsize=64000;
  } else {
    vocabsize=vocab->num_words();
    if (!sent_start_sym.size()) sent_start_idx=-1;
    else if (!(sent_start_idx=vocab->word_index(sent_start_sym))) {
      fprintf(stderr,"No sentence start symbol %s in vocabulary, exit.\n", sent_start_sym.c_str());
      exit(-1);
    }
  }

  while (fscanf(in,MAX_WLEN_FMT_STRING,charbuf)!=EOF) {
    //if (num_read % 1000000 == 0) fprintf(stderr,"Read %lld words\n",num_read);
    if (grow_vocab) idx=vocab->add_word(charbuf);
    else idx=vocab->word_index(charbuf);

    //fprintf(stderr, "%s ",charbuf);
    
    if (fscanf(in,MAX_WLEN_FMT_STRING,charbuf) == EOF) {
      fprintf(stderr,"Expected a logprob after the word %s", charbuf);
      exit(-1);
    }

    //fprintf(stderr,"Word %s = %d\n", charbuf, idx);

    if (idx==sent_start_idx) v.clear();

    if (v.size()<read_order) { 
      v.push_back(idx); 
    }
    else v.back()=idx;

    if (v.size()==read_order) {
      
      //if (this->average) den=this->GetCount(v);
      //else {
      if (read_order == 1) den=m_uni_counts_den;
      else den=this->GetCount(read_order-1, &v[0]);
      //}
      prob = exp(atof(charbuf))/(CT) den;

      //fprintf(stderr,"%f / %f = %f\n", exp(atof(charbuf)), den, prob);
      bool flag=false;
      for (int i=1;i<read_order;i++) {
        if (v[i]==sent_start_idx) {
          flag=true;
        }
      }
      if (!flag) this->IncrementBackoff(read_order,&v[0],prob);
      this->IncrementProb(v, prob);
      
      //fprintf(stderr, "%s %f\n",charbuf,prob);
    }
    //fprintf(stderr,"Cadd ");print_indices(stderr,v);fprintf(stderr," %d\n", 1);

    if (v.size()==read_order) 
      for (int j=0;j<read_order-1;j++) 
	v[j]=v[j+1];
    
    num_read++;
  }
  fprintf(stderr,"Finished reading %ld words.\n", num_read);

  return(num_read);
}

/**
 * The function below assumes it will be called after the
 * counts from the text have been read in and populated.
**/
template <typename KT, typename CT>
long MultiOrderProbs<KT, CT>::InitializeProbsFromStorage(Storage_t<KT, CT> *data, const int read_order, const int sent_start_idx) {
  long num_read=0;
  KT idx;
  std::vector<KT> v;
  CT prob= 0.0, den;
  //fprintf(stderr,"Order %i\n", read_order);
  for (size_t di=0;di<data->size(); di++) {
    num_read++;
    idx=data->data(di);
    if (idx==sent_start_idx) v.clear();

    if (v.size()<read_order) { 
      v.push_back(idx); 
    }
    else v.back()=idx;

    if (v.size()==read_order) {
      
      //if (this->average) den=this->GetCount(v);
      //else {
      if (read_order == 1) den=m_uni_counts_den;
      else den=this->GetCount(read_order-1, &v[0]);
      //}
      prob = data->prob(di)/(CT) den;

      //fprintf(stderr,"%f / %f = %f\n", data->prob(di), den, prob);
      bool flag=false;
      for (int i=1;i<read_order;i++) {
        if (v[i]==sent_start_idx) {
          flag=true;
        }
      }
      if (!flag) this->IncrementBackoff(read_order,&v[0],prob);
      this->IncrementProb(v, prob);
      
      //fprintf(stderr, "%f %f\n",data->prob(di),prob);
    }
    //fprintf(stderr,"Cadd ");print_indices(stderr,v);fprintf(stderr," %d\n", 1);

    if (v.size()==read_order) 
      for (int j=0;j<read_order-1;j++) 
	v[j]=v[j+1];
    
  }
  fprintf(stderr,"Finished reading %ld words.\n", num_read);

  return(num_read);
}

template <typename KT, typename CT>
CT MultiOrderProbs<KT, CT>::Increment_wo_del(
  struct matrix *m, const KT *indices, const CT value) {

  indextype idx=FindEntry(m, (byte *) indices, 1);
  CT *valp = (CT *) &(m->data[idx*m->size_of_entry]);
  *valp+=value;
  return(*valp);
}

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::SetProb(const int order, const KT *v, 
					  CT value) {
  allocate_matrices_probs(order);
  m_probs[order]->setvalue(v,value);
}

template <typename KT, typename CT>
float MultiOrderProbs<KT, CT>::GetProb(const int order, const KT *v) {
  if (order >= m_probs.size()) return(0);
  return(m_probs[order]->getvalue(v));
}

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::SetCount(const int order, const KT *v, 
					  const CT value) {
  allocate_matrices_counts(order);
  m_counts[order]->setvalue(v,value);
}

template <typename KT, typename CT>
CT MultiOrderProbs<KT, CT>::GetCount(const int order, const KT *v) {
  if (order >= m_counts.size()) return(0);
  return(m_counts[order]->getvalue(v));
}

template <typename KT, typename CT>
CT MultiOrderProbs<KT, CT>::IncrementCount(const int order, const KT *v, 
					    const CT value) {
  allocate_matrices_counts(order);
  return(Increment_wo_del(m_counts[order]->m, v, value));
}

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::IncrementProb(const int order, const KT *v, 
					    const CT value) {
  allocate_matrices_probs(order);
  Increment_wo_del(m_probs[order]->m, v, value);
}

template <typename KT, typename CT>
void *MultiOrderProbs<KT, CT>::
StepCountsOrder(const bool init, const int order, KT *indices, CT *value) {
  if (order>=m_counts.size()) return(NULL);
  return(m_counts[order]->stepthrough(init,indices,value));
}

template <typename KT, typename CT>
void *MultiOrderProbs<KT, CT>::OrderedStepCountsOrder(const bool init, 
			const int order, KT *indices, CT *value){
  if (order>=m_counts.size()) return(NULL);
  return(m_counts[order]->ordered_stepthrough(init,indices,value));
}

template <typename KT, typename CT>
CT MultiOrderProbs<KT, CT>::GetBackoff(const std::vector<KT> &v) {
  return(GetBackoff(v.size(),&v[0]));
}

template <typename KT, typename CT>
MultiOrderProbs<KT, CT>::~MultiOrderProbs() {
  for (int i=1;i<m_counts.size();i++) 
    if (std::find(m_do_not_delete.begin(),m_do_not_delete.end(),i) 
	== m_do_not_delete.end()) // The matrix was internally malloced
      delete m_counts[i];
  for (int i=1;i<m_probs.size();i++)
    delete m_probs[i];
  for (int i=1;i<m_backoffs.size();i++)
    delete m_backoffs[i];

}

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::allocate_matrices_counts(int o) {
  if (o<m_counts.size()) return;
  if (vocabsize==0) {
    fprintf(stderr,"MultiOrderCounts: Please set a reasonable vocabulary size. Exit.\n");
    exit(-1);
  }
  if (hashsize==0) hashsize=600000;
  indextype real_hashsize;
  
  int old_size=m_counts.size();
  m_counts.resize(o+1,NULL);
  for (int i=std::max(1,old_size);i<m_counts.size();i++) {
    assert(m_counts[i]==NULL);
    std::vector<KT> v(i,(KT) vocabsize);
    // Some heuristics to try to get reasonable hash sizes. Not too well tested
    real_hashsize=std::min(std::max(1000,(indextype) (vocabsize*pow((float) i,3))),hashsize);
    //fprintf(stderr,"min(%d*%d^3=%.0f,", vocabsize,i, vocabsize*pow((float) i, 3));
    //fprintf(stderr,"%d)\n", (int) hashsize);
    if (i>4 && order_size(i-1)>1) real_hashsize=order_size(i-1)*2+1;
    //if (i>2) fprintf(stderr,"Allocating counts matrices for order %d, size %d (prev size %d, vocabsize %d)\n", i, real_hashsize, order_size(i-1), this->vocabsize);
    m_counts[i]= new sikMatrix<KT, CT>(i,real_hashsize,0);
  }
}

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::allocate_matrices_probs(int o) {
  if (o<m_probs.size()) return;
  if (vocabsize==0) {
    fprintf(stderr,"MultiOrderCounts: Please set a reasonable vocabulary size. Exit.\n");
    exit(-1);
  }
  if (hashsize==0) hashsize=600000;
  indextype real_hashsize;
  
  int old_size=m_probs.size();
  m_probs.resize(o+1,NULL);
  for (int i=std::max(1,old_size);i<m_probs.size();i++) {
    assert(m_probs[i]==NULL);
    std::vector<KT> v(i,(KT) vocabsize);
    // Some heuristics to try to get reasonable hash sizes. Not too well tested
    real_hashsize=std::min(std::max(1000,(indextype) (vocabsize*pow((float) i,3))),hashsize);
    //fprintf(stderr,"min(%d*%d^3=%.0f,", vocabsize,i, vocabsize*pow((float) i, 3));
    //fprintf(stderr,"%d)\n", (int) hashsize);
    if (i>4 && order_size(i-1)>1) real_hashsize=order_size(i-1)*2+1;
    //if (i>2) fprintf(stderr,"Allocating counts matrices for order %d, size %d (prev size %d, vocabsize %d)\n", i, real_hashsize, order_size(i-1), this->vocabsize);
    m_probs[i]= new sikMatrix<KT, float>(i,real_hashsize,0);
  }
}

template <typename KT, typename CT>
bool MultiOrderProbs<KT, CT>::NextVector(std::vector<KT> &v) {
  if (m_cur_ng>=m_counts[m_cur_order]->num_entries()) {
    m_cur_ng=0;
    m_cur_order++;
    while (m_cur_order<m_counts.size() &&
           m_counts[m_cur_order]->num_entries()==0)
      m_cur_order++;
    if (m_cur_order>=m_counts.size()) {
      m_cur_order=1;
      m_cur_ng=0;
      return(false);
    }
  }
  v.resize(m_cur_order);
  const KT *keys=m_counts[m_cur_order]->Idx2Keyp(m_cur_ng);
  for (int i=0;i<m_cur_order;i++) {
    v[i]=keys[i];
  }
  m_cur_ng++;
  return(true);
}


template <typename KT, typename CT>
CT MultiOrderProbs<KT, CT>::IncrementCountCache(
  const int order, const KT *indices, const CT value) {
  allocate_matrices_counts(order);
  CT *v;
  c_cache.resize(c_cache.size()+1);
  c_cache_t &c=c_cache.back();

  c.order=order;
  c.val=value;

  struct matrix *m = m_counts[order]->m;
  const indextype idx=FindEntry(m, (byte *) indices, 1);
  c.index=idx;

  v=m_counts[order]->Idx2Valp(idx); //(CT *) &(m->data[idx*m->size_of_entry]);
  *v += value;
  return(*v);
}

template <typename KT, typename CT>
CT MultiOrderProbs<KT, CT>::IncrementProbCache(
  const int order, const KT *indices, const CT value) {
  allocate_matrices_probs(order);
  CT *v;
  p_cache.resize(p_cache.size()+1);
  p_cache_t &p=p_cache.back();

  p.order=order;
  p.val=value;

  struct matrix *m = m_probs[order]->m;
  const indextype idx=FindEntry(m, (byte *) indices, 1);
  p.index=idx;

  v=m_probs[order]->Idx2Valp(idx); //(CT *) &(m->data[idx*m->size_of_entry]);
  *v += value;
  return(*v);
}
template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::
IncrementBackoffCache(const int order, const KT *indices, const CT value) {
  allocate_matrices_backoffs(order);
  CT *v;
  bo_cache.resize(bo_cache.size()+1);
  bo_cache_t &b=bo_cache.back();
  
  b.order=order;
  b.bo=value;

  struct matrix *m = m_backoffs[order]->m;
  const indextype idx=FindEntry(m, (byte *) indices, 1);
  b.index=idx;

  v=(CT *) &(m->data[idx*m->size_of_entry]);
  *v += value;
}

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::ResetCaches() {
  c_cache.resize(0);
  p_cache.resize(0);
  bo_cache.resize(0);

  this->min_cc_cache.resize(this->m_counts.size()+1);
  for (int i=1;i<this->m_counts.size();i++) {
    this->min_cc_cache[i]=this->m_counts[i]->num_entries();
    //fprintf(stderr,"cc[%d]=%d\n",i,m_counts[i]->num_entries);
  }
  this->min_cc_cache[this->m_counts.size()]=0;

  this->min_p_cache.resize(this->m_probs.size()+1);
  for (int i=1;i<this->m_probs.size();i++) {
    this->min_p_cache[i]=this->m_probs[i]->num_entries();
    //fprintf(stderr,"cc[%d]=%d\n",i,m_counts[i]->num_entries);
  }
  this->min_p_cache[this->m_probs.size()]=0;

  this->min_bo_cache.resize(m_backoffs.size()+1);
  for (int i=2;i<m_backoffs.size();i++) {
    this->min_bo_cache[i]=this->m_backoffs[i]->num_entries();
    //fprintf(stderr,"bo[%d]=%d\n",i,m_backoffs[i]->num_entries);
  }
  this->min_bo_cache[m_backoffs.size()]=0;
 }

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::UndoCached() {
  /* This could be speeded up by assuming that all cached 
     values are of the same order
  */
  for (long i=this->c_cache.size()-1;i>=0;i--) {
    struct MultiOrderProbs<KT,CT>::c_cache_t &c=this->c_cache[i];
    struct matrix *m = this->m_counts[c.order]->m;
    * (CT *)(&(m->data[c.index*m->size_of_entry])) -= c.val;
  }

  for (int j=1;j<this->m_counts.size();j++) {
    for (long i=this->m_counts[j]->num_entries()-1;i>=this->min_cc_cache[j];i--) {
      RemoveEntryIdx(MultiOrderProbs<KT, CT>::m_counts[j]->m,i);
    }
  }

  for (long i=this->p_cache.size()-1;i>=0;i--) {
    struct MultiOrderProbs<KT,CT>::p_cache_t &p=this->p_cache[i];
    struct matrix *m = this->m_probs[p.order]->m;
    * (CT *)(&(m->data[p.index*m->size_of_entry])) -= p.val;
  }

  for (int j=1;j<this->m_probs.size();j++) {
    for (long i=this->m_probs[j]->num_entries()-1;i>=this->min_p_cache[j];i--) {
      RemoveEntryIdx(MultiOrderProbs<KT, CT>::m_probs[j]->m,i);
    }
  }

  /* Copy of the beginning of the func modified for backoffs.. */
  for (long i=this->bo_cache.size()-1;i>=0;i--) {
    struct MultiOrderProbs<KT,CT>::bo_cache_t &c=this->bo_cache[i];
    struct matrix *m = m_backoffs[c.order]->m;
    * (CT *) (&(m->data[c.index*m->size_of_entry])) -= c.bo;
  }

  for (int j=2;j<m_backoffs.size();j++) { 
    for (long i=m_backoffs[j]->num_entries()-1;i>=min_bo_cache[j];i--) 
      RemoveEntryIdx(MultiOrderProbs<KT, CT>::m_backoffs[j]->m,i);
  }
}


template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::
IncrementBackoff(const int order, const KT *v, const CT *value) {
  if (order==1) {
    m_uni_bo+= *value;
    return;
  }
  allocate_matrices_backoffs(order);
  m_backoffs[order]->increment(v,value);
}

template <typename KT, typename CT>
CT MultiOrderProbs<KT, CT>::GetBackoff(const int order, const KT *v) {
  if (order==1) return(m_uni_bo);
  if (order>=m_backoffs.size()) return(0);
  return(m_backoffs[order]->getvalue(v));
}


template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::
IncrementBackoff(const std::vector<KT> &v, const CT *value) {
  IncrementBackoff(v.size(),&v[0],value);
}

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::SetBackoff(
  const int order, const KT *v, const CT *bo) {
  allocate_matrices_backoffs(order);
  if (order>1) {
    m_backoffs[order]->setvalue(v,bo);
    return;
  }
  m_uni_bo = *bo;
}

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::SetBackoff(
  const std::vector<KT> &v, const CT *bo) {
  SetBackoff(v.size()+1,&v[0],bo);
}

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::GetBackoff(const int order, 
					    const KT *v, CT *value){
  if (order>=m_backoffs.size()) {
    *value=(CT) 0;
    return;
  }
  if (order>1) {
    m_backoffs[order]->getvalue(v,value);
    return;
  }
  memcpy(value,&m_uni_bo,sizeof(CT));
}

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::
IncrementBackoff(const int order, const KT *indices, const CT den) {
  allocate_matrices_backoffs(order);
  if (order>1) {
    struct matrix *m = m_backoffs[order]->m;
    indextype idx=FindEntry(m, (byte *) indices,1);
    CT *bop=(CT *)(&(m->data[idx*m->size_of_entry]));
    *bop+=den;
    // Delete node, if matches default value
    if (!memcmp(bop,m->default_value, m->size_of_entry)) RemoveEntryIdx(m,idx);
    return;
  }
  m_uni_bo+=den;
}

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::allocate_matrices_backoffs(int o
) {
  if (o<m_backoffs.size()) return;
  if (this->vocabsize==0) {
    fprintf(stderr,"MultiOrderCounts_t: Please set a reasonable vocabulary size. Exit.\n");
    exit(-1);
  }
  if (this->hashsize==0) this->hashsize=300000;
  indextype real_hashsize;

  int old_size=m_backoffs.size();
  m_backoffs.resize(o+1,NULL);
  for (int i=std::max(2,old_size);i<m_backoffs.size();i++) {
    assert(m_backoffs[i]==NULL);
    // Some heuristics to try to get reasonable hash sizes. Not too well tested
    real_hashsize=std::min(std::max(1000,(indextype) (this->vocabsize*pow((float) i,3))),this->hashsize);
    if (i>4 && bo_order_size(i-1)>1) real_hashsize=bo_order_size(i-1)*2+1;
    fprintf(stderr,"Allocating backoff matrices for order %d, size %ld", i, (long) real_hashsize);
    if (i>2) fprintf(stderr,"(prev_size %d, vocabsize %d)\n", bo_order_size(i-1), this->vocabsize);
    else fprintf(stderr,"\n");
    m_backoffs[i]=new sikMatrix<KT, CT>(i-1,real_hashsize,0);
    fprintf(stderr,"allocation succesful\n");
  }
}

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::clear_derived_counts() {
  m_uni_bo=0;
  this->m_counts[1]->clear();
  for (int i=2; i<this->m_counts.size()-1;i++) {
    this->m_counts[i]->clear();
    m_backoffs[i]->clear();
  }
  m_backoffs.back()->clear();
}

template <typename KT, typename CT>
void *MultiOrderProbs<KT, CT>::
StepBackoffsOrder(const bool init, const int order, KT *indices, CT *value) {
  if (order>=MultiOrderProbs<KT, CT>::m_counts.size()) return(NULL);
  assert(order>=2);
  return(m_backoffs[order]->stepthrough(init,indices,value));
}

template <typename KT, typename CT>
void MultiOrderProbs<KT, CT>::WriteProbs(FILE *out) {
  std::vector<KT> v;
  CT val, prob;
  CT bo_val;

  fprintf(out,"\\vocabsize %d\n", MultiOrderProbs<KT, CT>::vocabsize);
  for (int o=1; o<=order(); o++) {
    v.resize(o);
    fprintf(out,"\\%d-gram counts\n",o);
    this->StepCountsOrder(true, o, &v[0], &val);
    while (this->StepCountsOrder(false, o, &v[0], &val)) {
      if (val==0) continue;
      print_indices(out, v);
      fprintf(out," ");
      prob=this->GetProb(o, &v[0]);
      this->write_prob(out, prob);
      fprintf(out,"\n");
      //fprintf(out, "%.12f %.12f\n", prob ,val);
    }

    v.resize(o-1);
    fprintf(out, "\n\\%d-gram backoffs\n",o);
    if (o==1) {
      fprintf(out, "[ ] ");
      this->write_prob(out, m_uni_bo);
      fprintf(out, "\n\n");
      continue;
    }
    StepBackoffsOrder(true, o, &v[0], &bo_val);
    while (StepBackoffsOrder(false, o, &v[0], &bo_val)) {
      if (bo_val==0) continue;
      print_indices(out, v);
      fprintf(out," ");
      this->write_prob(out, bo_val);
      fprintf(out,"\n");
    }
    fprintf(out,"\n");
  }
}
