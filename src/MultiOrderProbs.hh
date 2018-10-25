// Library for storing and modifying the n-gram counts
#ifndef MULTIORDERPROBS_HH
#define MULTIORDERPROBS_HH

#include <vector>
#include <algorithm>
#include <cassert>
#include "sikMatrix.hh"
#include "Storage.hh"

template <typename KT, typename CT>
class MultiOrderProbs {
public:
  MultiOrderProbs(): vocabsize(1000000), hashsize(0), m_cur_order(1), m_cur_ng(0), m_uni_counts_den((CT) 0), m_uni_bo((CT) 0) {}
  ~MultiOrderProbs();

  /* data structures to store probability, counts and backoffs */
  int vocabsize;
  indextype hashsize;
  CT m_uni_counts_den;
  bool average; 
  std::vector<sikMatrix<KT, CT> * > m_counts;   
  std::vector<sikMatrix<KT, double> * > m_probs;    
  std::vector<sikMatrix<KT, double> * > m_backoffs;    
 
  inline indextype order_size(int o){
    if (o<=order()) return(m_counts[o]->num_entries());
    return(0);
  }
  inline indextype bo_order_size(int o){
    if (o<m_backoffs.size()) return(m_backoffs[o]->num_entries());
    return(0);
  }

  /* Functions to read counts and probabilities from text */
  long InitializeCountsFromText(FILE *in, Vocabulary *vocab, const bool grow_vocab, const int read_order, const std::string &sent_start_sym);
  long InitializeProbsFromText(FILE *in, Vocabulary *vocab, const bool grow_vocab, const int read_order, const std::string &sent_start_sym);
  long InitializeCountsFromStorage(Storage_t<KT, CT> *data, const int read_order, const int sent_start_idx);
  long InitializeProbsFromStorage(Storage_t<KT, CT> *data, const int read_order, const int sent_start_idx);

  /* Functions to step through counts and backoff vectors  */
  //void UseAsCounts(sikMatrix<KT, CT> *mat);
  bool NextVector(std::vector<KT> &v);
  //void RandomVector(std::vector<KT> &v);
  //void RandomVector(std::vector<KT> &v, Storage_t<KT, CT> &data);

  inline void *StepCountsOrder(const bool init, const int order, 
				KT *indices, CT *value);

  inline void DeleteCurrentST(const int order) {
    m_counts[order]->delete_current_st();
  }
  inline void *OrderedStepCountsOrder(const bool init, const int order, 
				KT *indices, CT *value);
  
  void *StepBackoffsOrder(const bool init, const int order, KT *indices, CT *value);
  //inline void RemoveEmptyNodes(const int order) {RemoveEmptyNodes(order,0,0);}

  /* Manipulation of probs */
  inline double GetProb(const std::vector<KT> &v);
  inline double GetProb(const int order, const KT *v);
  inline void SetProb(const std::vector <KT> &v, const CT value) {
    SetProb(v.size(),&v[0],value);
  }
  inline void SetProb(const int order, const KT *v, const CT value);
  
  inline void IncrementProb(std::vector<KT> &v, const CT value) {
    IncrementProb(v.size(),&v[0],value);
  }
  inline void IncrementProb(const std::vector<KT> &v, const CT value);
  inline void IncrementProb(const int order, const KT *v,const CT value);
 


  /* Manipulation of counts */
  inline CT GetCount(const std::vector<KT> &v);
  inline CT GetCount(const int order, const KT *v);

  inline void SetCount(const std::vector <KT> &v, const CT value) {
    SetCount(v.size(),&v[0],value);
  }
  inline void SetCount(const int order, const KT *v, const CT value);

  inline CT IncrementCount(std::vector<KT> &v, const CT value) {
    return(IncrementCount(v.size(),&v[0],value));
  }
  inline CT IncrementCount(const std::vector<KT> &v, const CT value);
  inline CT IncrementCount(const int order, const KT *v, const CT value);
  
  /* Manipulation of backoffs */
  inline void SetBackoff(const int order, const KT *v, const CT *value);
  inline void SetBackoff(const std::vector <KT> &v, const CT *value);
  
  void GetBackoff(const int order, const KT *v, CT *value);
  inline CT GetBackoff(const std::vector<KT> &v);
  CT GetBackoff(const int order, const KT *v);
  
  void IncrementBackoff(const int order, const KT *v, const CT *value);
  void IncrementBackoff(const std::vector<KT> &v, const CT *value);
  void IncrementBackoff(const int order, const KT *v, const CT value);

  inline void RemoveDefaultBackoffs() {
    for (int o=order();o>=2;o--) {
      RemoveDefaultValues(MultiOrderProbs<KT, CT>::m_backoffs[o]->m);
    }
  }


  /* Cached manipulations */
  CT IncrementCountCache(const int order, const KT *v, 
				  const CT value);
  
  CT IncrementProbCache(const int order, const KT *v, 
				  const CT value);

  void IncrementBackoffCache(const int order, const KT *v, 
				       const CT value);

  void ResetCaches();  
  void UndoCached();

  /* Functions to help with reading and writing numbers */ 
  inline void write_num(FILE *out, const int val)   {fprintf(out,"%d",val);}
  inline void write_num(FILE *out, const unsigned int val)   {fprintf(out,"%u",val);}
  inline void write_num(FILE *out, const long val)   {fprintf(out,"%ld",val);}
  inline void write_num(FILE *out, const float val) {fprintf(out,"%.12f",val);}
  inline void write_num(FILE *out, const double val) {fprintf(out,"%.4f",val);}
  inline void write_prob(FILE *out, const float val) {fprintf(out,"%.12f",val);}
  inline void read_num(int *val, const std::string *s, bool *ok) {
    *val = str::str2long(s, ok);
  }

  inline void read_num(long *val, const std::string *s, bool *ok) {
    *val = str::str2long(s, ok);
  }

  inline void read_num(unsigned int *val, const std::string *s, bool *ok) {
    *val = str::str2long(s, ok);
  }

  inline void read_num(float *val, const std::string *s, bool *ok) {
    *val = str::str2float(s, ok);
  }
  inline void read_num(double *val, const std::string *s, bool *ok) {
    *val = str::str2float(s, ok);
  }

  inline int order() {return MultiOrderProbs<KT, CT>::m_counts.size()-1;}
  void WriteProbs(FILE *out);
  void RemoveOrder(int order);
  //void RemoveEmptyNodes(const int order, const indextype start, const indextype bo_start);
  void clear_derived_counts();
 
protected:
 
  CT m_uni_bo;
  
  std::vector<int> m_do_not_delete;
  void allocate_matrices_counts(int o);
  void allocate_matrices_probs(int o);
  void allocate_matrices_backoffs(int o);
  
  /* For NextVector() */
  int m_cur_order;
  indextype m_cur_ng;

  /* Low-level matrix library functions */
  CT Increment_wo_del(struct matrix *m, const KT *indices, const CT value);

  /* Cache data structures for counts, probs and backoffs */
  struct c_cache_t {
    int order;
    CT val;
    indextype index;
  };

  std::vector<c_cache_t> c_cache;
  std::vector<indextype> min_cc_cache;
  
  struct p_cache_t {
    int order;
    CT val;
    indextype index;
  };

  std::vector<p_cache_t> p_cache;
  std::vector<indextype> min_p_cache;
  
  struct bo_cache_t {
    int order;
    CT bo;
    indextype index;
  };

  std::vector<bo_cache_t> bo_cache;
  std::vector<indextype> min_bo_cache;
};


#include "MultiOrderProbs_tmpl.hh"
#endif
