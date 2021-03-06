// This program creates first creates a full  unpruned language model of 
// given order. The model can be then pruned, if some options are given.
#include "conf.hh"
#include "io.hh"
#include "ProbsLM.hh"

int main(int argc, char **argv) {
  conf::Config config;
  config("Usage: probs2arpa [OPTIONS] textin arpa\nConverts probabilities to counts format from given textprob.\n")
    ('n', "norder=INT","arg","0","Maximal order included in the model (default unrestricted)")
    ('B', "vocabin=FILE", "arg", "", "Restrict the vocabulary to given words")
    ('H', "hashsize=INT", "arg", "0", "Size of the reserved hash table. Speed vs. memory consumption.")
    ('s', "smallvocab", "", "", "Vocabulary is less than 65000 entries. Saves some memory.")
    ('S', "smallmem", "", "", "Do not cache the data. Saves some memory, but slows down.")
    ('C', "clear_history", "", "", "Clear LM history on each start of sentence tag (<s>).")
    ('p',"prunetreshold","arg","-1","Prune out the n-grams, for which the score does not exceed the threshold. Default: no pruning=0.")
    ('L', "longint", "", "", "Store counts in a long int type. Needed for big training sets.")
    ('O', "cutoffs=\"val1 val2 ... valN\"", "arg", "", "Use the specified cutoffs. The last value is used for all higher order n-grams.")
    ('U', "write_vocab=FILE", "arg", "", "Write resulting vocabulary FILE.")
    ('N', "discard_unks", "", "", "Remove n-grams containing OOV words.")
    ('a', "average", "", "", "Average probability instead of dividing by backoff counts.")
    ('P', "readprob=FILE","arg","", "Read the specified file that has probability associated with each word.\nEach line of the textprob has a word and an associated probability separated by space.");
  config.parse(argc,argv,2);

  const int n=config["norder"].get_int();
  const indextype hashs=config["hashsize"].get_int();
  const bool smallvocab=config["smallvocab"].specified;
  const bool smallmem=config["smallmem"].specified;
  const float prunetreshold=config["prunetreshold"].get_double();
  const bool discard_unks=config["discard_unks"].specified;
  const bool longint=config["longint"].specified;
  const bool average=config["average"].specified;
  const std::string readprob(config["readprob"].get_str());
  const std::string vocabout(config["write_vocab"].get_str());

  
  bool ok=true;
  std::vector<int> cutoffs=str::long_vec<int>(config["cutoffs"].get_str(),&ok);
  if (!ok) {
    fprintf(stderr,"Error parsing cutoffs, exit\n");
    exit(-1);
  }

  io::Stream::verbose=true;
  std::string dataname=config.arguments.at(0);
  if (dataname=="-") {
    fprintf(stderr,"probs2arpa might need to scan the input several times.  Cannot read the main data from stdin \"-\". Exit.\n");
    exit(-1);
  }

  std::string arpaout=config.arguments.at(1);
  std::string ss_sym;
  if (config["clear_history"].specified) {
    fprintf(stderr,"clearing history\n");
    ss_sym="<s>";
  }

  try {
    /* Construct the base kn-smoothed model */
    fprintf(stderr,"Estimating counts\n");
    ProbsLM *problm=NULL;
    TreeGram lm;
    bool init_disc=true;

    /* Parse the arguments, create the right kind of model*/
    if (!smallvocab) {
      problm=new ProbsLM_impl<int, double>(dataname, "", n, NULL, ss_sym, readprob, average, hashs);
    } else {
      problm=new ProbsLM_impl<unsigned short, double>(dataname, "", n, NULL, ss_sym, readprob, average, hashs);
    } 
    
    fprintf(stderr, "The model will use ");
    if (smallvocab) fprintf(stderr,"small vocabulary (<65534) and ");
    else fprintf(stderr,"large vocabulary and ");
    fprintf(stderr,"absolute discounting.\n");

    problm->create_model(std::max((float) 0.0,prunetreshold));
    
    //Write out the arpa
    io::Stream out(arpaout, "w");
    problm->counts2lm(out.file);
    out.close();
    
  }
  catch (std::exception &e) {
    fprintf(stderr,"%s\n",e.what());
    exit(1); 
  }
}
