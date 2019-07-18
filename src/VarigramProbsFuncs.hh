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
  virtual void write(FILE *out)=0;
  virtual void set_clear_symbol(std::string s)=0;
  virtual void set_discard_unks(bool x)=0;
  void write_vocab(FILE *out) {m_vocab->write(out);}

  inline void write_file(std::string lmname) { io::Stream out(lmname, "w"); write(out.file); out.close(); }

protected:
  double m_datacost_scale;
  double m_datacost_scale2;
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
  void write(FILE *out);

  inline void set_clear_symbol(std::string s) {
    assert(m_problm);
    m_data->clear_lm_history=m_vocab->word_index(s);
    if (!m_data->clear_lm_history) {
      fprintf(stderr,"No \"<s>\" in history, --clear_history cannot be used. Exit.\n");
      exit(-1);
    }
    m_problm->set_sentence_boundary_symbol(s);
  }
  void set_discard_unks(bool x) {
    m_problm->discard_ngrams_with_unk=x;
  }


private:
  ProbsLM_t<KT, ICT> *m_problm;
  NgramCounts_t<KT, ICT> *m_initial_ng;

  bool reestimate_with_history(std::vector<KT> &history);
  double modify_model(std::map<KT, ICT> &new_c, std::map<KT, ICT> &new_prob, const std::vector<KT> &v, const double ml_norm);
  Storage_t<KT, ICT> *m_data;
  void prune();
};

#include "VarigramProbsFuncs_tmpl.hh"
#endif
