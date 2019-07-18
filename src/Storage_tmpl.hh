// Library for storing corpuses in memory
#include "def.hh"
#include "str.hh"
#include "io.hh"

template <typename T>
void Storage<T>::read(FILE *in, Vocabulary &voc) {
  char buf[MAX_WLEN+1];
  data_vec.reserve(100000);
  //fprintf(stderr,"storage read:");
  while (fscanf(in,MAX_WLEN_FMT_STRING,buf)==1) {
    data_vec.push_back(voc.word_index(buf));
    //fprintf(stderr," %s(%d)", buf, voc.word_index(buf));
  }
  //fprintf(stderr,"\n");
}

template <typename T, typename ICT>
void Storage_t<T, ICT>::read_prob(FILE *in, Vocabulary &voc) {
  char buf[MAX_WLEN+1];
  this->data_vec.reserve(100000);
  prob_vec.reserve(100000);
  
  while (fscanf(in,MAX_WLEN_FMT_STRING,buf)==1) {
    this->data_vec.push_back(voc.word_index(buf));

    if (fscanf(in,MAX_WLEN_FMT_STRING,buf)==EOF) {
      fprintf(stderr,"Storage::read_prob Expected a logprob after the word %s", buf);
      exit(-1);
    }
    prob_vec.push_back(atof(buf));
  }
}

template <typename T, typename ICT>
void Storage_t<T, ICT>::read_topk_probs(FILE *in, Vocabulary &voc, int k, int sent_start_idx) {
  char buf[MAX_WLEN+1];
  this->data_vec.reserve(100000);
  prob_vec.reserve(100000);
  topk_prob_vec.reserve(100000);

  while (fscanf(in,MAX_WLEN_FMT_STRING,buf)==1) {
    T word = voc.word_index(buf);
    this->data_vec.push_back(word);

    if (fscanf(in,MAX_WLEN_FMT_STRING,buf)==EOF) {
      fprintf(stderr,"Storage::read_topk_probs: Expected a logprob %s", buf);
      exit(-1);
    }
    prob_vec.push_back(atof(buf));
    std::vector<std::pair<T,ICT> > topk;

    if (word != sent_start_idx) {
      for (int i=0; i<k; i++){
        if (fscanf(in,MAX_WLEN_FMT_STRING,buf)==EOF) {
          fprintf(stderr,"Storage::read_topk_probs: Expected a word %s", buf);
          exit(-1);
        }
        word = voc.word_index(buf);

        if (fscanf(in,MAX_WLEN_FMT_STRING,buf)==EOF) {
          fprintf(stderr,"Storage::read_topk_probs: Expected a logprob %s", buf);
          exit(-1);
        }
        ICT logprob = atof(buf);
     
        std::pair<T,ICT> item = std::make_pair(word, logprob);
        topk.push_back(item);      
      }
    }

    topk_prob_vec.push_back(topk);
    //fprintf(stderr," %s(%d)", buf, voc.word_index(buf));
  }
  //fprintf(stderr,"\n");
}

template <typename T, typename ICT>
void Storage_t<T, ICT>::initialize_fast_search_lists(
  const int order, sikMatrix<T, ICT> *refmat, sikMatrix<T, ICT> *curmat) {
  assert(refmat!=NULL);
  m_refmat=refmat;  
  m_lists.clear();
  this->m_lists2.clear();
  m_last_init_mapped=(m_refmat->dims<=2);
  if (m_last_init_mapped) m_lists.resize(m_refmat->num_entries());
  else this->m_lists2.resize(m_refmat->num_entries());
  std::vector<T> v;
  size_t i;

  if (this->clear_lm_history==-1) {
    for (i=0;i<order-1;i++) v.push_back(this->data_vec[i]);
    v.push_back(0); // Make the vector right size
    for (;i<this->size();i++) {
      const indextype idx = FindEntry(m_refmat->m, (byte *) (&v[0]), 0);
      v[order-1]=this->data_vec[i];
      if (idx>=0)  // The prefix exists ?
	if (!curmat || FindEntry(curmat->m, (byte *) (&v[0]), 0)==-1) {
	  //fprintf(stderr,"Adding to list ");print_indices(v); fprintf(stderr,": %ld", curmat);
	  //if (curmat) fprintf(stderr," %d", curmat->getvalue(&v[0]));
	  //fprintf(stderr,"\n");
	  if (m_last_init_mapped)  m_lists[idx][this->data_vec[i]]++;
	  else this->m_lists2[idx].push_back(this->data_vec[i]);
	}
      for (int j=0;j<order-1;j++) v[j]=v[j+1];
    }
    return;
  }

  for (size_t i=0;i<this->size();i++) {
    if (this->data_vec[i]==this->clear_lm_history) {
      v.clear();
    }
    
    if (v.size()<order) 
      v.push_back(this->data_vec[i]);
    else
      v[order-1]=this->data_vec[i];
    
    if (v.size()==order) {
      const indextype idx = FindEntry(m_refmat->m, (byte *) (&v[0]), 0);
      if (idx>=0) { // The prefix exists ?
	// This is for growing-pruning iterations, slows down otherwise
	if (!curmat || FindEntry(curmat->m, (byte *) (&v[0]), 0)==-1) {
	  //fprintf(stderr,"adding %d", data_vec[i]);
	  //if (curmat) fprintf(stderr,"(%d)", curmat->getvalue(&v[0]));
	  //fprintf(stderr,"\n");
	  if (m_last_init_mapped) m_lists[idx][this->data_vec[i]]++;
	  else this->m_lists2[idx].push_back(this->data_vec[i]);
	}
      }
      for (int j=0;j<order-1;j++) v[j]=v[j+1];
    }
  }
  return;
}

template <typename T, typename ICT>
void Storage_t<T, ICT>::initialize_fast_search_lists_topk_probs(
  const int order, sikMatrix<T, ICT> *refmat, sikMatrix<T, ICT> *curmat, const int k) {
  assert(refmat!=NULL);
  m_refmat=refmat;  
  m_lists.clear();
  m_probs.clear();
  this->m_lists2.clear();
  this->m_probs2.clear();
  this->m_check2.clear();
  fprintf(stderr,"initialize_fast_search_lists_probs: %i\n", order);
  m_last_init_mapped=(m_refmat->dims<=2);
  if (m_last_init_mapped) {
    m_lists.resize(m_refmat->num_entries());
    m_probs.resize(m_refmat->num_entries()); 
  } else {
    //fprintf(stderr,"initialize_fast_search_lists_probs: %d\n", m_refmat->num_entries());
    this->m_lists2.resize(m_refmat->num_entries());
    this->m_probs2.resize(m_refmat->num_entries());
    this->m_check2.resize(m_refmat->num_entries());
  }
  std::vector<T> v;
  size_t i;

  if (this->clear_lm_history==-1) {
    for (i=0;i<order-1;i++) v.push_back(this->data_vec[i]);
    v.push_back(0); // Make the vector right size
    for (;i<this->size();i++) {
      const indextype idx = FindEntry(m_refmat->m, (byte *) (&v[0]), 0);
      v[order-1]=this->data_vec[i];
      if (idx>=0) { // The prefix exists ?
	if (!curmat || FindEntry(curmat->m, (byte *) (&v[0]), 0)==-1) {
	  //fprintf(stderr,"Adding to list ");print_indices(v); fprintf(stderr,": %ld", curmat);
	  //if (curmat) fprintf(stderr," %d", curmat->getvalue(&v[0]));
	  //fprintf(stderr,"\n");
	  if (m_last_init_mapped)  {
            m_lists[idx][this->data_vec[i]]++;
            m_probs[idx][this->data_vec[i]] += prob(i);
          } else {
            this->m_lists2[idx].push_back(this->data_vec[i]);
            this->m_check2[idx].push_back(true);
            this->m_probs2[idx].push_back(prob(i));
          }
	}
        std::vector<std::pair<T,ICT> > topk_probs_for_idx = topk_probs(i);
        for (int l=0; l<k; l++) {
          v[order-1]=topk_probs_for_idx[l].first;
          if (!curmat || FindEntry(curmat->m, (byte *) (&v[0]), 0)==-1) {
            if (m_last_init_mapped)  {
              m_probs[idx][v[order-1]] += exp(topk_probs_for_idx[l].second);
            } else {
              this->m_lists2[idx].push_back(v[order-1]);
              this->m_check2[idx].push_back(false);
              this->m_probs2[idx].push_back(exp(topk_probs_for_idx[l].second));
            }
          }
        }
      }
      v[order-1]=this->data_vec[i];
      for (int j=0;j<order-1;j++) v[j]=v[j+1];
    }
    return;
  }

  for (size_t i=0;i<this->size();i++) {
    if (this->data_vec[i]==this->clear_lm_history) {
      v.clear();
    }
    
    if (v.size()<order) 
      v.push_back(this->data_vec[i]);
    else
      v[order-1]=this->data_vec[i];
    
    if (v.size()==order) {
      const indextype idx = FindEntry(m_refmat->m, (byte *) (&v[0]), 0);
      if (idx>=0) { // The prefix exists ?
	// This is for growing-pruning iterations, slows down otherwise
	if (!curmat || FindEntry(curmat->m, (byte *) (&v[0]), 0)==-1) {
	  //fprintf(stderr,"adding %d", data_vec[i]);
	  //if (curmat) fprintf(stderr,"(%d)", curmat->getvalue(&v[0]));
	  //fprintf(stderr,"\n");
          if (m_last_init_mapped)  {
            m_lists[idx][this->data_vec[i]]++;
            m_probs[idx][this->data_vec[i]] += prob(i);
          } else {
            this->m_lists2[idx].push_back(this->data_vec[i]);
            this->m_check2[idx].push_back(true);
            this->m_probs2[idx].push_back(prob(i));
          }
	}
        std::vector<std::pair<T,ICT> > topk_probs_for_idx = topk_probs(i);
        for (int l=0; l<k; l++) {
          v[order-1]=topk_probs_for_idx[l].first;
          if (!curmat || FindEntry(curmat->m, (byte *) (&v[0]), 0)==-1) {
            if (m_last_init_mapped)  {
              m_probs[idx][v[order-1]] += exp(topk_probs_for_idx[l].second);
            } else {
              this->m_lists2[idx].push_back(v[order-1]);
              this->m_check2[idx].push_back(false);
              this->m_probs2[idx].push_back(exp(topk_probs_for_idx[l].second));
            }
          }
        }
      }
      v[order-1]=this->data_vec[i];
      for (int j=0;j<order-1;j++) v[j]=v[j+1];
    }
  }
  return;
}

template <typename T, typename ICT>
void Storage_t<T, ICT>::initialize_fast_search_lists_probs(
  const int order, sikMatrix<T, ICT> *refmat, sikMatrix<T, ICT> *curmat) {
  assert(refmat!=NULL);
  m_refmat=refmat;  
  m_lists.clear();
  m_probs.clear();
  this->m_lists2.clear();
  this->m_probs2.clear();
  fprintf(stderr,"initialize_fast_search_lists_probs: %i\n", order);
  m_last_init_mapped=(m_refmat->dims<=2);
  if (m_last_init_mapped) {
    m_lists.resize(m_refmat->num_entries());
    m_probs.resize(m_refmat->num_entries()); 
  } else {
    //fprintf(stderr,"initialize_fast_search_lists_probs: %d\n", m_refmat->num_entries());
    this->m_lists2.resize(m_refmat->num_entries());
    this->m_probs2.resize(m_refmat->num_entries());
  }
  std::vector<T> v;
  size_t i;

  if (this->clear_lm_history==-1) {
    for (i=0;i<order-1;i++) v.push_back(this->data_vec[i]);
    v.push_back(0); // Make the vector right size
    for (;i<this->size();i++) {
      const indextype idx = FindEntry(m_refmat->m, (byte *) (&v[0]), 0);
      v[order-1]=this->data_vec[i];
      if (idx>=0)  // The prefix exists ?
	if (!curmat || FindEntry(curmat->m, (byte *) (&v[0]), 0)==-1) {
	  //fprintf(stderr,"Adding to list ");print_indices(v); fprintf(stderr,": %ld", curmat);
	  //if (curmat) fprintf(stderr," %d", curmat->getvalue(&v[0]));
	  //fprintf(stderr,"\n");
	  if (m_last_init_mapped)  {
            m_lists[idx][this->data_vec[i]]++;
            m_probs[idx][this->data_vec[i]] += prob(i);
          } else {
            this->m_lists2[idx].push_back(this->data_vec[i]);
            this->m_probs2[idx].push_back(prob(i));
          }
	}
      for (int j=0;j<order-1;j++) v[j]=v[j+1];
    }
    return;
  }

  for (size_t i=0;i<this->size();i++) {
    if (this->data_vec[i]==this->clear_lm_history) {
      v.clear();
    }
    
    if (v.size()<order) 
      v.push_back(this->data_vec[i]);
    else
      v[order-1]=this->data_vec[i];
    
    if (v.size()==order) {
      const indextype idx = FindEntry(m_refmat->m, (byte *) (&v[0]), 0);
      if (idx>=0) { // The prefix exists ?
	// This is for growing-pruning iterations, slows down otherwise
	if (!curmat || FindEntry(curmat->m, (byte *) (&v[0]), 0)==-1) {
	  //fprintf(stderr,"adding %d", data_vec[i]);
	  //if (curmat) fprintf(stderr,"(%d)", curmat->getvalue(&v[0]));
	  //fprintf(stderr,"\n");
          if (m_last_init_mapped)  {
            m_lists[idx][this->data_vec[i]]++;
            m_probs[idx][this->data_vec[i]] += prob(i);
          } else {
            this->m_lists2[idx].push_back(this->data_vec[i]);
            this->m_probs2[idx].push_back(prob(i));
          }
	}
      }
      for (int j=0;j<order-1;j++) v[j]=v[j+1];
    }
  }
  return;
}

template <typename T, typename ICT>
void Storage_t<T, ICT>::initialize_fast_search_lists_for_pruning(
  const int order, sikMatrix<T, ICT> *refmat) {
  assert(refmat!=NULL);
  m_refmat=refmat;  
  m_last_init_mapped=false;
  m_lists.clear();
  this->m_lists2.clear();
  this->prune_lists.clear();
  this->prune_lists.resize(m_refmat->num_entries(),0);
  std::vector<T> v;

  for (indextype i=0;i<this->size();i++) {
    if (this->data_vec[i]==this->clear_lm_history) {
      v.clear();
    }
    
    if (v.size()<order) 
      v.push_back(this->data_vec[i]);
    else
      v[order-1]=this->data_vec[i];
    
    if (v.size()==order) {
      const indextype idx = FindEntry(m_refmat->m, (byte *) (&v[0]), 0);
      if (idx>=0) this->prune_lists[idx]+=1;
      for (int j=0;j<order-1;j++) v[j]=v[j+1];
    }
  }
  return;
}

template <typename T, typename ICT>
void Storage_t<T, ICT>::init_fsl_file(
  const int order, sikMatrix<T, ICT> *refmat, std::string &fname, 
  Vocabulary *voc) {

  assert(refmat!=NULL);
  m_refmat=refmat;  
  m_last_init_mapped=(m_refmat->dims<=2);
  m_lists.clear();
  this->m_lists2.clear();

  if (m_last_init_mapped) m_lists.resize(m_refmat->num_entries());
  else this->m_lists2.resize(m_refmat->num_entries());

  io::Stream in(fname,"r",io::REOPENABLE);
  std::vector<T> v;
  char sbuf[MAX_WLEN];

  if (this->clear_lm_history==-1) {
    for (int i=0;i<order-1;i++) {
      if (fscanf(in.file,"%s",sbuf)!=1) return;
      v.push_back(voc->word_index(sbuf));
    }
    v.push_back(0); //vector is of right size
    
    while (fscanf(in.file,"%s",sbuf)==1) {
      indextype idx = FindEntry(m_refmat->m, (byte *) (&v[0]), 0);
      //fprintf(stderr,"%d: idx %d(<%d) ",i,idx,m_lists.size());
      T widx=voc->word_index(sbuf);
      v[order-1]=widx;
      if (idx>=0) { // The prefix exists ?
	if (m_last_init_mapped) m_lists[idx][widx]++;
	else this->m_lists2[idx].push_back(widx);
      }
      for (int j=0;j<order-1;j++) v[j]=v[j+1];
    }
  } else {
    while (fscanf(in.file,"%s",sbuf)==1) {
      T widx=voc->word_index(sbuf);
      if (widx==this->clear_lm_history) v.clear();
      if (v.size()<order) 
	v.push_back(widx);
      else
	v[order-1]=widx;

      if (v.size()==order) {
	indextype idx = FindEntry(m_refmat->m, (byte *) (&v[0]), 0);
	//fprintf(stderr,"%d: idx %d(<%d) ",i,idx,m_lists.size());
	if (idx>=0) { // The prefix exists ?
	  if (m_last_init_mapped) m_lists[idx][widx]++;
	  else this->m_lists2[idx].push_back(widx);
	}
	for (int j=0;j<order-1;j++) v[j]=v[j+1];
      }
    }
  }
}

template <typename T, typename ICT>
void Storage_t<T, ICT>::fast_search_next(std::vector<T> *v, int *ridx, ICT *rval) {
  *ridx=-1;

  /* if v!=NULL, init new search */
  if (v!=NULL) {
    //fprintf(stderr,"a%d",(*v)[0]);
    m_cur_vec=FindEntry(m_refmat->m,(byte *) (&(*v)[0]),0);
    //fprintf(stderr,"MC %d(%ld)\n",m_cur_vec,m_lists[m_cur_vec].size());
    if (m_last_init_mapped) {
      // FIXME: Why is the next if clause needed ?
      if (m_cur_vec >= m_lists.size()) {
	m_cur_vec = -1;
	return;
      }
      m_cur_vec_idx=m_lists[m_cur_vec].begin();
    }
    else this->m_cur_vec_idx2=0;
    return;
  }
  //fprintf(stderr,"b%d_%d",m_cur_vec_idx,m_lists[m_cur_vec].size());
  
  if (m_cur_vec == -1) return;
  if (m_last_init_mapped && (m_cur_vec>= m_lists.size() ||
			     m_cur_vec_idx == m_lists[m_cur_vec].end()))
    return;
  if (!m_last_init_mapped && ( m_cur_vec>= this->m_lists2.size() ||
			       this->m_cur_vec_idx2>=this->m_lists2[m_cur_vec].size()))
    return;

  if (m_last_init_mapped) {
    *ridx = m_cur_vec_idx->first;
    *rval = m_cur_vec_idx->second;
    m_cur_vec_idx++;
  } else {
    *ridx=this->m_lists2[m_cur_vec][this->m_cur_vec_idx2++];
    *rval=1;
  }
  return ;
}

template <typename T, typename ICT>
void Storage_t<T, ICT>::fast_search_next(std::vector<T> *v, int *ridx, ICT *rval, ICT *rprob) {
  *ridx=-1;

  /* if v!=NULL, init new search */
  if (v!=NULL) {
    //fprintf(stderr,"a%d",(*v)[0]);
    m_cur_vec=FindEntry(m_refmat->m,(byte *) (&(*v)[0]),0);
    //fprintf(stderr,"MC %d(%ld)\n",m_cur_vec,m_lists[m_cur_vec].size());
    if (m_last_init_mapped) {
      // FIXME: Why is the next if clause needed ?
      if (m_cur_vec >= m_lists.size()) {
	m_cur_vec = -1;
	return;
      }
      m_cur_vec_idx=m_lists[m_cur_vec].begin();
      m_cur_vec_prob=m_probs[m_cur_vec].begin();
    }
    else this->m_cur_vec_idx2=0;
    return;
  }
  //fprintf(stderr,"b%d_%d",m_cur_vec_idx,m_lists[m_cur_vec].size());
  
  if (m_cur_vec == -1) return;
  if (m_last_init_mapped && (m_cur_vec>= m_lists.size() ||
			     m_cur_vec_idx == m_lists[m_cur_vec].end()))
    return;
  if (!m_last_init_mapped && ( m_cur_vec>= this->m_lists2.size() ||
			       this->m_cur_vec_idx2>=this->m_lists2[m_cur_vec].size()))
    return;

  if (m_last_init_mapped) {
    *ridx=m_cur_vec_idx->first;
    *rval=m_cur_vec_idx->second;
    *rprob=m_cur_vec_prob->second;
    m_cur_vec_idx++;
    m_cur_vec_prob++;
  } else {
    *ridx=this->m_lists2[m_cur_vec][this->m_cur_vec_idx2];
    *rval=1;
    *rprob=this->m_probs2[m_cur_vec][this->m_cur_vec_idx2++];

  }
  return ;
}

template <typename T, typename ICT>
void Storage_t<T, ICT>::fast_search_next2(std::vector<T> *v, int *ridx, ICT *rval, ICT *rprob) {
  *ridx=-1;

  /* if v!=NULL, init new search */
  if (v!=NULL) {
    //fprintf(stderr,"a%d",(*v)[0]);
    m_cur_vec=FindEntry(m_refmat->m,(byte *) (&(*v)[0]),0);
    //fprintf(stderr,"MC %d(%ld)\n",m_cur_vec,m_lists[m_cur_vec].size());
    if (m_last_init_mapped) {
      // FIXME: Why is the next if clause needed ?
      if (m_cur_vec >= m_lists.size()) {
	m_cur_vec = -1;
	return;
      }
      m_cur_vec_idx=m_lists[m_cur_vec].begin();
      m_cur_vec_prob=m_probs[m_cur_vec].begin();
    }
    else this->m_cur_vec_idx2=0;
    return;
  }
  //fprintf(stderr,"b%d_%d",m_cur_vec_idx,m_lists[m_cur_vec].size());
  
  if (m_cur_vec == -1) return;
  if (m_last_init_mapped && (m_cur_vec>= m_lists.size() ||
			     m_cur_vec_idx == m_lists[m_cur_vec].end()))
    return;
  if (!m_last_init_mapped && ( m_cur_vec>= this->m_lists2.size() ||
			       this->m_cur_vec_idx2>=this->m_lists2[m_cur_vec].size()))
    return;

  if (m_last_init_mapped) {
    *ridx=m_cur_vec_idx->first;
    *rval=m_cur_vec_idx->second;
    *rprob=m_cur_vec_prob->second;
    m_cur_vec_idx++;
    m_cur_vec_prob++;
  } else {
    *ridx=this->m_lists2[m_cur_vec][this->m_cur_vec_idx2];
    *rval=1;
    if (!this->m_check2[m_cur_vec][this->m_cur_vec_idx2])
      *rval=0;
    *rprob=this->m_probs2[m_cur_vec][this->m_cur_vec_idx2++];

  }
  return ;
}
