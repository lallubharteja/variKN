// Functions for the n-gram growing algorithm
#ifndef VARIGRAMFUNCS_HH
#define VARIGRAMFUNCS_HH

#include "ProbsLM.hh"
#include <map>

class VarigramProbs {
public:
  inline VarigramProbs() :
  m_datacost_scale(1),
  m_datacost_scale2(0),
  m_ngram_prune_target(0),
  m_max_order(INT_MAX)
  { }

  virtual ~VarigramProbs() {}
  inline void set_datacost_scale(double f) {m_datacost_scale=f;}
  inline void set_datacost_scale2(double f) {m_datacost_scale2=f;}
  inline void set_ngram_prune_target(indextype i) {m_ngram_prune_target=i;}
  inline void set_max_order(int i) {m_max_order=i;}
  
  virtual void initialize(std::string infilename, indextype hashsize, 
			  std::string clhist, std::string readprob, 
			  std::string vocabname="")=0;
  virtual void grow(int iter2_lim=1)=0;
  //virtual void write_narpa(FILE *out)=0;
  //virtual void write_debug_counts(FILE *out)=0;
  //virtual void write_probs_as_counts(FILE *out)=0;
  virtual void write(FILE *out, bool arpa)=0;
  virtual void set_clear_symbol(std::string s)=0;
  //virtual void set_zeroprobgrams(bool)=0;
  //virtual void set_cutoffs(std::vector<int> v)=0;
  virtual void set_discard_unks(bool x)=0;
  //virtual void set_all_discounts(float x)=0;
  //bool absolute;
  void write_vocab(FILE *out) {m_vocab->write(out);}

  inline void write_file(std::string lmname, bool arpa) { io::Stream out(lmname, "w"); write(out.file, arpa); out.close(); }

protected:
  float m_datacost_scale;
  float m_datacost_scale2;
  indextype m_ngram_prune_target;
  int m_max_order;
  std::string m_infilename;
  Vocabulary *m_vocab;
};

template <typename KT, typename ICT>
class VarigramProbs_t : public VarigramProbs {
public:
  VarigramProbs_t():
  VarigramProbs(),
  m_problm(NULL),
  m_initial_ng(NULL),
  m_data(NULL)
  { }


  ~VarigramProbs_t();
  void initialize(std::string infilename, indextype hashsize,
		  std::string clhist, std::string readprob,
		  std::string vocabname="");
  void grow(int iter2_lim=1);
  //inline void write_narpa(FILE *out) {m_problm->counts2asciilm(out);}
  //inline void write_debug_counts(FILE *out) {m_problm->counts2ascii(out);}
  //inline void write_probs_as_counts(FILE *out) {m_problm->probs2ascii(out);}
  //inline void set_zeroprobgrams(bool x) {m_problm->zeroprobgrams=x;}
  void write(FILE *out, bool arpa);

  inline void set_clear_symbol(std::string s) {
    assert(m_problm);
    m_data->clear_lm_history=m_vocab->word_index(s);
    if (!m_data->clear_lm_history) {
      fprintf(stderr,"No \"<s>\" in history, --clear_history cannot be used. Exit.\n");
      exit(-1);
    }
    m_problm->set_sentence_boundary_symbol(s);
  }
  /*void set_cutoffs(std::vector<int> v) {
    m_problm->cutoffs=v;
  }*/
  void set_discard_unks(bool x) {
    m_problm->discard_ngrams_with_unk=x;
  }

  //void set_all_discounts(float x) {
  //  m_problm->init_disc(x);
  //}

private:
  ProbsLM_t<KT, ICT> *m_problm;
  NgramCounts_t<KT, ICT> *m_initial_ng;

  bool reestimate_with_history(std::vector<KT> &history);
  double modify_model(std::map<KT, ICT> &new_c, std::map<KT, ICT> &new_prob, const std::vector<KT> &v, const float ml_norm);
  Storage_t<KT, ICT> *m_data;
  //void get_unigram_counts(std::string &infilename, int ndrop, int nfirst, int *type);
  //void get_unigram_counts(std::string &infilename, int ndrop, int nfirst, unsigned short *type);

  //void printmatrix_bo(sikMatrix<KT, typename MultiOrderCounts<KT, ICT>::bo_3c> *m);
  //void printmatrix_bo(sikMatrix<KT, typename MultiOrderCounts<KT, ICT>::bo_c> *m);

  void prune();
};

#include "VarigramProbsFuncs_tmpl.hh"
#endif
