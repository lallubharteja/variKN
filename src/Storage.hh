// Library for storing corpuses in memory
#ifndef STORAGE_HH
#define STORAGE_HH

#include "Vocabulary.hh"
#include "sikMatrix.hh"

template <typename T>
class Storage {
public:
  Storage() : clear_lm_history(-1) {}
  void read(FILE *in, Vocabulary &voc);
  std::vector<T> datavec; 
  inline T data(size_t idx) const {return datavec[idx];}
  inline size_t size() const {return datavec.size();}
  int clear_lm_history; // Negative values: disabled

protected:
  std::vector<std::vector<T> > m_lists2;
  //indextype m_cur_vec;
  size_t m_cur_vec_idx2;
  //bool m_last_init_mapped;
};

template <typename T, typename ICT>
class Storage_t : public Storage<T> {
public:
  void read_prob(FILE *in, Vocabulary &voc);
  inline ICT prob(size_t idx) const {return exp(probvec[idx]);}
  void initialize_fast_search_lists(
    const int order, sikMatrix<T, ICT> *refmat, 
    sikMatrix<T, ICT> *curmat);
  void initialize_fast_search_lists_probs(
    const int order, sikMatrix<T, ICT> *refmat, 
    sikMatrix<T, ICT> *curmat);
  void initialize_fast_search_lists_for_pruning(
    const int order, sikMatrix<T, ICT> *refmat);
  void init_fsl_file(const int order, sikMatrix<T, ICT> *refmat,
		     std::string &fname, Vocabulary *voc);
  void fast_search_next(std::vector<T> *v, int *ridx, ICT *rval);
  void fast_search_next(std::vector<T> *v, int *ridx, ICT *rval, ICT *rprob);
  std::vector<ICT> prune_lists;

private:
  std::vector<std::map<T, ICT> > m_lists;
  std::vector<std::map<T, ICT> > m_probs;
  std::vector<std::vector<ICT> > m_probs2;
  typename std::map<T, ICT>::iterator m_cur_vec_idx;
  typename std::map<T, ICT>::iterator m_cur_vec_prob;
  sikMatrix<T, ICT> *m_refmat;
  std::vector<ICT> probvec;

  // These are already declared in the parent, why the need to redclare here?
  indextype m_cur_vec;
  bool m_last_init_mapped;
};

#include "Storage_tmpl.hh"
#endif
