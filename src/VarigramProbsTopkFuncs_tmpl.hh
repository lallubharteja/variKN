// Functions for the n-gram growing algorithm

//// DEBUG FUNCTIONS 
/*
template <typename KT, typename ICT>
void VarigramProbs_t<KT, ICT>::printmatrix_bo(sikMatrix<KT, typename MultiOrderProbs<KT, ICT>::bo_3c> *m) {
  typename MultiOrderProbs<KT, ICT>::bo_3c *value;
  for (indextype i=0;i<m->num_entries();i++) {
    print_indices(m->Idx2Keyp(i), m->dims);
    value=m->Idx2Valp(i);
    fprintf(stderr,"=%d(%d %d %d)\n",value->den, value->nzer[0], value->nzer[1], value->nzer[2]);
  }
}

template <typename KT, typename ICT>
void VarigramProbs_t<KT, ICT>::printmatrix_bo(sikMatrix<KT, typename MultiOrderProbs<KT, ICT>::bo_c> *m) {
  std::vector<KT> idx(m->dims);
  typename MultiOrderProbs<KT, ICT>::bo_c value;
  m->stepthrough(true, &idx[0], &value);
  while (m->stepthrough(false,&idx[0],&value)) {
    print_indices(idx);
    fprintf(stderr,"=%d(%d)\n",value.den, value.nzer);
  }
}
*/
///////////////

template <typename KT, typename ICT>
VarigramProbsTopk_t<KT, ICT>::~VarigramProbsTopk_t() {
  delete m_problm;
  delete m_initial_ng;
  delete m_data;
}

template <typename KT, typename ICT>
void VarigramProbsTopk_t<KT, ICT>::
initialize(std::string infilename, indextype hashsize, 
	   std::string clhist, std::string readprob,
	   std::string vocabname, int k) {
  io::Stream in;
  io::Stream::verbose=true;
  m_data=new Storage_t<KT, ICT>;
  m_k=k;
  Storage_t<KT, ICT> *datastorage_tmp;
  datastorage_tmp=m_data;
  
  fprintf(stderr,"Creating unigram model, ");
  if (hashsize) fprintf(stderr,"hashize %d, ", hashsize);
  /* Construct the unigram model, choose ther right template arguments*/
  fprintf(stderr, "using ProbLM...\n");
  m_problm = new ProbsLM_impl<KT, ICT>(infilename, vocabname, 1, datastorage_tmp, clhist, readprob, false, k, hashsize);
  
  //m_problm->init_disc(0.71);  
  m_vocab=&(m_problm->vocab);
  //m_problm->use_ehist_pruning(m_data->size());
  fprintf(stderr,"Initialization for 1st iteration done\n");

  m_infilename=infilename;

  if (clhist.size()) set_clear_symbol(clhist);
}

template <typename KT, typename ICT>
void VarigramProbsTopk_t<KT, ICT>::grow(int iter2_lim) {
  std::vector<KT> new_history;
  std::vector<int> accepted_mem;
  int tot_acc=0;
  int iter=0, iter2=0;
  int cur_order;
  int accepted;

  std::vector<sikMatrix<KT, ICT> *> *sik_c;
  sik_c=&(m_problm->mop->m_counts);
  
  //int update_coeff_counter=1;
  while (iter2<iter2_lim) {
    int old_hist_size=-1;
    while (true) {
      accepted=0;
      iter++;
      while (true) {
	if (!m_problm->MopNextVector(new_history) 
	    || new_history.size()>=m_max_order)
	  goto LAST;

	if (new_history.size()!=old_hist_size) {
	  //update_coeff_counter++;
	  cur_order=new_history.size();
	  sikMatrix<KT, ICT> *curref1=(*sik_c)[new_history.size()], *curref2=NULL; 
	  if (iter2>0) 
	    curref2=(*sik_c)[new_history.size()];
	  m_data->initialize_fast_search_lists_topk_probs(new_history.size()+1, curref1, curref2, m_k);
	}
	old_hist_size=new_history.size();

	if (reestimate_with_history(new_history)) {
	  accepted++;
	  //if (update_coeff_counter) update_coeff_counter++;
	  //if (!(accepted%1000)) {
	  //  fprintf(stderr,"New:");
	  //  for (size_t j=0;j<new_history.size();j++) 
	  //    fprintf(stderr," %s",m_problm->vocab.word(new_history[j]).c_str());
	  //  fprintf(stderr,"\n");
	  //}
	} //else fprintf(stderr,".");

      }
      accepted_mem.push_back(accepted);
      tot_acc+=accepted;
      for (int i=0;i<accepted_mem.size();i++) 
	fprintf(stderr,"Round %d: accepted %d\n",i,accepted_mem[i]);
    }
  LAST:
    //for (int i=1;i<=m_problm->order();i++) {
    //  fprintf(stderr,"matrix order %d:\n",i);
    //  m_problm->print_matrix(i);
    //}
    //if (old_hist_size >0) {
    //  m_problm->set_leaveoneout_discounts(old_hist_size);
    //}

    // Ugly fix follows, make this look nicer....
    //if (m_problm->get_sentence_boundary_symbol()>0) {
    //  m_problm->remove_sent_start_prob();
    //}

    prune();
    tot_acc+=accepted;
    fprintf(stderr,"%d iterations, %d accepted\n", iter,tot_acc);
    iter2++;
  }
}

template <typename KT, typename ICT>
void VarigramProbsTopk_t<KT, ICT>::prune() {
  if (m_ngram_prune_target) {
    double cur_scale = m_datacost_scale;

    int round = 0;
    indextype prev_num_grams = m_problm->num_grams();
    double prev_scale = cur_scale * 2;

    while (double(m_problm->num_grams()) > double(m_ngram_prune_target) * 1.03) {

        if (round == 0) {
          fprintf(stderr, "Currently %d ngrams. First prune with E=D=%.5f\n", m_problm->num_grams(), cur_scale);
          m_problm->prune_model((ICT) cur_scale, m_data);
          ++round;
          continue;
      }

      double scale_diff = cur_scale - prev_scale;
      indextype gram_diff = prev_num_grams - m_problm->num_grams();

      fprintf(stderr, "Previous round increased E from %.4f to %.4f and this pruned the model from %d to %d ngrams\n", prev_scale, cur_scale, prev_num_grams, m_problm->num_grams());
      fprintf(stderr, "I still need to remove %d grams\n", m_problm->num_grams() - m_ngram_prune_target);

      double increase = (double)(m_problm->num_grams() - m_ngram_prune_target) / (double)(gram_diff);
      fprintf(stderr, "Without limits I would increase E with %.4f (which is %.4f %%) to %.4f\n", increase*scale_diff, (increase*scale_diff)/cur_scale, cur_scale+(increase*scale_diff));

      prev_scale = cur_scale;
      prev_num_grams = m_problm->num_grams();

      cur_scale = std::max(std::min(cur_scale+(increase*scale_diff), cur_scale * 1.5), cur_scale * 1.05);

      fprintf(stderr, "With limits I increase E with %.4f (which is %.4f %%) to %.4f\n", cur_scale - prev_scale, (cur_scale-prev_scale)/prev_scale, cur_scale);

      m_problm->prune_model(cur_scale, m_data);
    }

    fprintf(stderr, "Finally, %d grams, which is %.4f %% off target\n", m_problm->num_grams(), (double)(m_ngram_prune_target-m_problm->num_grams())/(double)(m_ngram_prune_target));
    if (double(m_problm->num_grams()) < double(m_ngram_prune_target) * 0.97) {
      fprintf(stderr, "WARNING: we pruned a bit too much! Increase D and run model training again to get the desired amount of n-grams\n");
    }
  } else {
    m_problm->prune_model(m_datacost_scale2, m_data);
  }
}

template <typename KT, typename ICT>
bool VarigramProbsTopk_t<KT, ICT>::reestimate_with_history(std::vector<KT> &v) {
  /* Modify the counts for the new history */
  bool accepted;
  int idx;
  ICT val, prob;
  ICT new_c_sum=0;
  std::map<KT, ICT> m_new_c;
  std::map<KT, ICT> m_new_prob;

#if 0
  fprintf(stderr,"WHIST: ");print_indices(stderr, v); //fprintf(stderr,"\n");
  fprintf(stderr," [");
  for (int i=0;i<v.size();i++) {
    fprintf(stderr," %s", m_vocab->word(v[i]).c_str());
  }
  fprintf(stderr,"]\n");
#endif

  /* Collect relevant statistics */
  m_data->fast_search_next2(&v, &idx, &val, &prob);
  m_data->fast_search_next2(NULL, &idx, &val, &prob);
  while (idx>=0) {
    m_new_c[idx]+=val;
    m_new_prob[idx]+=prob;
    new_c_sum += val;
    m_data->fast_search_next2(NULL, &idx, &val, &prob);
  }
  
  if (m_new_c.size()==0) {
    //fprintf(stderr,"No hits found\n");
    return(false);
  }
  
  m_problm->MopResetCaches();
  //m_problm->print_matrix(v.size());
  double delta1=modify_model(m_new_c,m_new_prob,v,1.0/new_c_sum);
  double delta2=m_new_c.size()*m_problm->model_cost_scale;

  const long size=m_problm->num_grams();
  const long origsize=size-m_new_c.size();
  delta2+=size*log2(size)-origsize*log2(origsize);
  delta2*=m_datacost_scale;

  double delta=delta1+delta2;

  //fprintf(stderr,"sizes %ld %ld\n", (long) m_new_c.size(), (long) origsize);
  //fprintf(stderr,"scales %g %g\n", m_datacost_scale, m_problm->model_cost_scale);
  //fprintf(stderr,"delta %.1f+%.1f=%.1f",delta1,delta2,delta);

  //fprintf(stderr,"hiorder mod: ");
  //m_problm->print_matrix(v.size()+1);
  if (delta<0 /*&& delta1 < -10*m_datacost_scale*/) {
    //fprintf(stderr,"\t Accepted.\n");
    accepted=true; 
  } else {
    //fprintf(stderr,"\tRejected.\n");
    m_problm->MopUndoCached();
    accepted=false;
  }
  //fprintf(stderr,"hiorder : ");
  //m_problm->print_matrix(v.size()+1);
  return accepted;
}

template <typename KT, typename ICT>
double VarigramProbsTopk_t<KT, ICT>::modify_model(
  std::map<KT,ICT> &m, std::map<KT,ICT> &m_prob, const std::vector<KT> &indices, const double ml_norm) {
  ICT val;
  int order=indices.size()+1;
  double logprobdelta=0.0;

  /* Check, if m_problm->m_order should be increased */
  if (m_problm->order() <order) m_problm->set_order(order);
  
  std::vector<KT> ind(indices);
  ind.resize(ind.size()+1);

  /* Get the current coding cost */

  //fprintf(stderr,"MAPPED %d\n", m.size());
  double ml_safelogprob=0.0;

  typename std::map<KT,ICT>::iterator it = m.begin();
  for ( ; it != m.end(); it++ ) {
    ind.back()=it->first;
    //fprintf(stderr,"Probbing ");print_indices(ind);fprintf(stderr,"\n");
    if (it->second != 0){
      logprobdelta+=safelogprob(m_problm->tableprob(ind))*it->second;
      // ml safelogprob enables earlier pruning and makes things slightly 
      // faster
      //fprintf(stderr,"prob %g ml_norm %g count %d\n", m_prob[it->first],  ml_norm, it->second);
      ml_safelogprob+=safelogprob(m_prob[it->first]* ml_norm)*it->second;
    }
  }

  const long origsize=m_problm->num_grams();
  const long size=origsize+m.size();
  //fprintf(stderr,"lpdelta %g + ml_safelogpro %g, +dscale1 %g *( mc %g + sdelta %zd + s1 %g-s2 %g\n", logprobdelta, -ml_safelogprob, m_datacost_scale, m_problm->model_cost_scale, m.size(), size*log2(size), origsize*log2(origsize));
  if ((logprobdelta-ml_safelogprob)+m_datacost_scale*(
      m_problm->model_cost_scale*m.size()+ size*log2(size)-origsize*log2(origsize))
      >=0 ) 
    return 1;
  
    for (it = m.begin(); it != m.end(); it++ ) {
      MultiOrderProbs<KT, ICT> *mop=m_problm->mop;
      ind.back()=it->first;
      val=it->second;
#if 0
      fprintf(stderr,"Counts %d\n", order);
      m_problm->print_matrix(order); 
      if (order>=2) {
      	fprintf(stderr,"Counts %d\n", order-1);
      	 m_problm->print_matrix(order-1);
      }
      //if (mop->m_backoffs.size()>order) {
      //fprintf(stderr,"backoffs %d\n", order);
//	printmatrix_bo(mop->m_backoffs[order]);
      //     }
      /* Increment the new higher order counts */
      fprintf(stderr,"Increment ");
      print_indices(ind);
      fprintf(stderr,"= %ld\n", (long) val);
#endif
      //Increment the new higher order counts
      ICT debug=mop->IncrementCountCache(order,&ind[0],val);
      assert(debug>=0);
      //Increment Backoff Cache with Prob
      mop->IncrementBackoffCache(order,&ind[0],m_prob[it->first]* ml_norm);
      //Increment Prob Cache
      mop->IncrementProbCache(order,&ind[0],m_prob[it->first]* ml_norm); 
    }
  /* Get the current coding cost */
  for (it = m.begin(); it != m.end(); it++ ) {
    ind.back()=it->first;
    logprobdelta-=safelogprob(m_problm->tableprob(ind))*it->second;
  }
  return(logprobdelta);
}

template <typename KT, typename ICT>
void VarigramProbsTopk_t<KT, ICT>::write(FILE *out) {
  m_problm->counts2lm(out);
}
