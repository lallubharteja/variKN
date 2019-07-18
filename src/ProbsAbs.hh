// The main library for the n-gram model estimation
#ifndef PROBSABS_HH
#define PROBSABS_HH

#include <algorithm>

#include <assert.h>
#include <math.h>

#include "Storage.hh"
#include "TreeGram.hh"
#include "MultiOrderCounts.hh"
#include "NgramCounts.hh"
#include "io.hh"

class ProbsAbs{
public:
  inline ProbsAbs(const std::string &dataname) : 
    zeroprobgrams(true), 
    discard_ngrams_with_unk(false),  m_order(0), 
    m_sent_boundary(-1), 
    m_data_name(dataname)
  {}
  virtual ~ProbsAbs() {}
  //virtual void create_model(float prunetreshold=-1.0)=0;
  virtual void probs2ascii(FILE* out)=0;
  
  inline int order() {return m_order;}
  //inline void use_ehist_pruning(int x) { m_ehist_estimate=x;}
  virtual void set_order(int o)=0;
  
  virtual void estimate_bo_counts()=0;
  virtual void estimate_nzer_counts()=0;
  //virtual void init_disc(float x)=0;
  virtual indextype num_grams()=0;
  
  /* Added for templatized func, too much casting otherwise */
  virtual void MocResetCaches()=0;
  virtual void MocUndoCached()=0;
  virtual indextype MocOrderSize(const int o)=0;
  virtual void write_counts(FILE *f)=0;
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
  
  typedef double disc;
 
  bool zeroprobgrams;
  std::string read_prob;
  bool average;
  
  virtual int debug() {return(-40);}
  virtual void print_matrix(int o) {}
  size_t input_data_size;

  //std::vector<int> cutoffs;
  /*inline int cutoff(int x) {
    if (!cutoffs.size()) return 0;
    if (x>cutoffs.size()) return cutoffs.back();
    assert(x>0);
    return cutoffs[x-1];
  }*/
  //bool discard_cutoffs;
  bool discard_ngrams_with_unk;
  double model_cost_scale;
protected:
  //NgramCounts *m_ng;
  int m_order;
  virtual inline void re_estimate_needed() {}
  int m_sent_boundary;
  //int m_ehist_estimate;
  std::string m_data_name;
  //virtual void initialize_minmax() =0;
  //std::vector<float> m_minvals, m_maxvals;
};

template <typename KT>
class ProbsAbs_k : public ProbsAbs {
public:
  ProbsAbs_k(const std::string &dataname):
    ProbsAbs(dataname) {}
  virtual ~ProbsAbs_k() { }
  virtual bool MocNextVector(std::vector<KT> &v)=0;
  virtual void remove_sent_start_prob()=0;
  
protected:
  inline bool KT_is_short(int *x) {return(false);}
  inline bool KT_is_short(unsigned short *x) {return(true);}
};

template <typename KT, typename CT>
class ProbsAbs_t : public ProbsAbs_k<KT> {
public:
  ProbsAbs_t(const std::string &datasource):
    ProbsAbs_k<KT>(datasource), m_eval_cache(NULL) {}
  ~ProbsAbs_t();
  void constructor_helper(
    const std::string &vocabname,
    const int order,
    const indextype hashsize, 
    const std::string &sent_boundary);
  virtual void estimate_bo_counts();
  virtual void init_probs(const int order, const std::string &sent_boundary);
  void probs2ascii(FILE* out);
  
  //void find_coeffs(float brak=-0.1, float precision=1e-3, float lin_precision=2e-2);
  
  double logprob_file(const char *name);
  double logprob_datastorage(const Storage<KT> &data);
  //double model_MDL_cost();
  void clear_lm_sentence_boundaries();
  
  MultiOrderCounts<KT, CT> *moc;
  virtual void add_zeroprob_grams()=0;
  
/* Added after templatization */
  inline bool MocNextVector(std::vector<KT> &v) {return moc->NextVector(v);}
  inline void MocUndoCached() {moc->UndoCached();}
  inline indextype MocOrderSize(const int o) {return moc->order_size(o);}
  inline void MocResetCaches() {moc->ResetCaches();}
  
  inline indextype num_grams() {
    indextype n_grams=0;
    for (int i=1;i<=moc->order();i++) {
      n_grams+=moc->order_size(i);
      //fprintf(stderr,"numg %d=%d\n",i,moc->order_size(i));
    }
    return(n_grams);
  }
  
  //void create_model(float prunetreshold);
  void remove_zeroprob_grams();
  //virtual void add_counts_for_backoffs() = 0;
  virtual void remove_sent_start_prob() {assert(false);}
  inline void write_counts(FILE *f) {moc->WriteCounts(f);}
  
protected:
  //inline float prob(const int order, const KT *i);
  //virtual float prob(const int order, const KT *i, const CT num)=0;
  //virtual float coeff(const int order, const KT *i)=0;

  //CT m_new_threshold;
  //CT m_ori_threshold; // Ugly hack...
  
  sikMatrix<float, float> *m_eval_cache;
  //NgramCounts_t<KT, CT> *m_ng_typed; // Just to avoid casting
  
  //virtual inline sikMatrix<KT, CT> *get_ct_matrix(int o, CT *foo, DT *bar) {assert(false);return(0);}
  //template <typename BOT> void add_counts_for_backoffs_fbase(BOT *);
  template <typename BOT> void add_zeroprob_grams_fbase(BOT *);
};

template <typename KT, typename ICT>
class ProbsAbs_int_disc: public ProbsAbs_t<KT, ICT> {
public:
  ProbsAbs_int_disc(
    const std::string data, const std::string vocab,
    const int order, 
    const std::string sent_boundary, const std::string read_prob, const bool average,
    const indextype hashsize=3000000);
  virtual void estimate_nzer_counts();
 
  //virtual float prob(const int order, const KT *i, const ICT num);
  //virtual float coeff(const int order, const KT *i);
  
  virtual inline void remove_sent_start_prob() {
    remove_sent_start_prob_fbase((MultiOrderCounts_counter_types::bo_c<ICT> *) NULL);
  }
  //virtual inline void add_counts_for_backoffs() {
  //  ProbsAbs_t<KT, ICT>::add_counts_for_backoffs_fbase((MultiOrderCounts_counter_types::bo_c<ICT> *) NULL);
  //}
  virtual inline void add_zeroprob_grams() {
    ProbsAbs_t<KT, ICT>::add_zeroprob_grams_fbase((MultiOrderCounts_counter_types::bo_c<ICT> *) NULL);
  }
  void print_matrix(int o) {if (o<this->moc->m_counts.size() && o>=1) this->moc->m_counts[o]->printmatrix();}

  void set_order(int o);
  //std::vector<InterKn::disc> m_discount;

protected:
  template <typename BOT> void remove_sent_start_prob_fbase(BOT *dummy);

  /*virtual void initialize_minmax() {
    this->m_minvals.resize(this->m_order,0.0);
    this->m_maxvals.resize(this->m_order,1.0);
  }*/
};

#include "ProbsAbs_tmpl.hh"
#endif
