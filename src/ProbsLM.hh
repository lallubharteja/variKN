// The main library for the n-gram model estimation
#ifndef PROBSLM_HH
#define PROBSLM_HH

#include <algorithm>

#include <assert.h>
#include <math.h>

#include "Storage.hh"
#include "TreeGram.hh"
#include "MultiOrderProbs.hh"
#include "NgramCounts.hh"
#include "io.hh"

class ProbsLM{
public:
  inline ProbsLM(const std::string &dataname) : 
    zeroprobgrams(true), 
    discard_ngrams_with_unk(false),  m_order(0), 
    m_sent_boundary(-1), 
    m_data_name(dataname)
  {}
  virtual ~ProbsLM() {}
  virtual void create_model(float prunetreshold=-1.0)=0;
  virtual void counts2lm(FILE *out)=0;
  virtual void probs2ascii(FILE *out)=0;
  
  inline int order() {return m_order;}
  virtual void set_order(int o)=0;
  
  virtual void estimate_bo_counts()=0;
  virtual indextype num_grams()=0;
  
  /* Added for templatized func, too much casting otherwise */
  virtual void MopResetCaches()=0;
  virtual void MopUndoCached()=0;
  virtual indextype MopOrderSize(const int o)=0;
  Vocabulary vocab;
  
  inline void set_sentence_boundary_symbol(std::string s) {
    m_sent_boundary=vocab.word_index(s);
    if (!m_sent_boundary) {
      fprintf(stderr,"No \"<s>\" in history, --clear_history cannot be used. Exit.\n");
      exit(-1);
    }
  }
  inline int get_sentence_boundary_symbol() {return(m_sent_boundary);}
  
  virtual void set_threshold(float x) {assert(false);}
  
  typedef float disc;
 
  bool zeroprobgrams;
  std::string read_prob;
  bool average;
  
  virtual int debug() {return(-40);}
  virtual void print_matrix(int o) {}
  size_t input_data_size;

  bool discard_ngrams_with_unk;
  float model_cost_scale;
protected:
  int m_order;
  virtual inline void re_estimate_needed() {}
  int m_sent_boundary;
  int m_ehist_estimate;
  std::string m_data_name;
};

template <typename KT>
class ProbsLM_k : public ProbsLM {
public:
  ProbsLM_k(const std::string &dataname):
    ProbsLM(dataname) {}
  virtual ~ProbsLM_k() { }
  virtual bool MopNextVector(std::vector<KT> &v)=0;
  virtual void remove_sent_start_prob()=0;
  
protected:
  inline bool KT_is_short(int *x) {return(false);}
  inline bool KT_is_short(unsigned short *x) {return(true);}
};

template <typename KT, typename CT>
class ProbsLM_t : public ProbsLM_k<KT> {
public:
  ProbsLM_t(const std::string &datasource):
    ProbsLM_k<KT>(datasource), m_eval_cache(NULL) {}
  ~ProbsLM_t();
  void constructor_helper(
    const std::string &vocabname,
    const int order, Storage_t<KT, CT> *datastorage, 
    const indextype hashsize, 
    const std::string &sent_boundary);
  virtual void estimate_bo_counts();
  virtual void init_probs(const int order, Storage_t<KT, CT> *datastorage, const std::string &sent_boundary);
  void probs2ascii(FILE *out);
  
  
  double logprob_file(const char *name);
  double logprob_datastorage(const Storage<KT> &data);
  void clear_lm_sentence_boundaries();
  void counts2lm(FILE *out);
  
  MultiOrderProbs<KT, CT> *mop;
  virtual void add_zeroprob_grams()=0;
  
/* Added after templatization */
  inline bool MopNextVector(std::vector<KT> &v) {return mop->NextVector(v);}
  inline void MopUndoCached() {mop->UndoCached();}
  inline indextype MopOrderSize(const int o) {return mop->order_size(o);}
  inline void MopResetCaches() {mop->ResetCaches();}
  
  inline indextype num_grams() {
    indextype n_grams=0;
    for (int i=1;i<=mop->order();i++) {
      n_grams+=mop->order_size(i);
      //fprintf(stderr,"numg %d=%d\n",i,mop->order_size(i));
    }
    return(n_grams);
  }
  
  void create_model(float prunetreshold);
  void remove_zeroprob_grams();
  virtual void remove_sent_start_prob() {assert(false);}
  virtual void prune_model(float threshold, Storage_t<KT, CT> *real_counts)=0;
  CT tableprob(std::vector<KT> &indices);
  
protected:
  virtual CT text_prob(const int order, const KT *i)=0;
  virtual CT text_coeff(const int order, const KT *i)=0;

    sikMatrix<float, float> *m_eval_cache;
  
  void add_zeroprob_grams_fbase(CT *);
  void prune_model_fbase(float threshold, Storage_t<KT, CT> *real_counts);
  virtual void prune_gram(std::vector<KT> &v, CT num) {assert(false);}
};

template <typename KT, typename ICT>
class ProbsLM_impl: public ProbsLM_t<KT, ICT> {
public:
  ProbsLM_impl(
    const std::string data, const std::string vocab,
    const int order, Storage_t<KT, ICT> *datastorage, 
    const std::string sent_boundary, const std::string read_prob, const bool average,
    const indextype hashsize=3000000);
  virtual inline void prune_model(float threshold, Storage_t<KT, ICT> *real_counts) {
    this->prune_model_fbase(threshold, real_counts);
  }
 
  virtual ICT text_prob(const int order, const KT *i);
  virtual ICT text_coeff(const int order, const KT *i);
  
  virtual inline void remove_sent_start_prob() {
    remove_sent_start_prob_fbase((ICT *) NULL);
  }

  virtual inline void add_zeroprob_grams() {
    ProbsLM_t<KT, ICT>::add_zeroprob_grams_fbase((ICT *) NULL);
  }
  void print_matrix(int o) {if (o<this->mop->m_counts.size() && o>=1) this->mop->m_counts[o]->printmatrix();}

  void set_order(int o);

protected:
  void remove_sent_start_prob_fbase(ICT *dummy);
  virtual void prune_gram(std::vector<KT> &v, ICT num);

};

#include "ProbsLM_tmpl.hh"
#endif
