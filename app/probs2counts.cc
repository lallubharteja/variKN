// This program creates first creates a full  unpruned language model of 
// given order. The model can be then pruned, if some options are given.
#include "conf.hh"
#include "io.hh"
#include "ProbsAbs.hh"

int main(int argc, char **argv) {
  conf::Config config;
  config("Usage: probs2counts [OPTIONS] textin\nConverts probabilities to counts format from given textprob.\n")
    ('n', "norder=INT","arg","0","Maximal order included in the model (default unrestricted)")
    ('B', "vocabin=FILE", "arg", "", "Restrict the vocabulary to given words")
    ('H', "hashsize=INT", "arg", "0", "Size of the reserved hash table. Speed vs. memory consumption.")
    ('s', "smallvocab", "", "", "Vocabulary is less than 65000 entries. Saves some memory.")
    ('S', "smallmem", "", "", "Do not cache the data. Saves some memory, but slows down.")
    ('C', "clear_history", "", "", "Clear LM history on each start of sentence tag (<s>).")
    ('L', "longint", "", "", "Store counts in a long int type. Needed for big training sets.")
    ('O', "cutoffs=\"val1 val2 ... valN\"", "arg", "", "Use the specified cutoffs. The last value is used for all higher order n-grams.")
    ('W', "write_counts=FILE", "arg", "", "Write resulting count matrices to FILE.")
    ('U', "write_vocab=FILE", "arg", "", "Write resulting vocabulary FILE.")
    ('N', "discard_unks", "", "", "Remove n-grams containing OOV words.")
    ('a', "average", "", "", "Average probability instead of dividing by backoff counts.")
    ('P', "readprob=FILE","arg","", "Read the specified file that has probability associated with each word.\nEach line of the textprob has a word and an associated probability separated by space.");
  config.parse(argc,argv,1);

  const int n=config["norder"].get_int();
  const indextype hashs=config["hashsize"].get_int();
  const bool smallvocab=config["smallvocab"].specified;
  const bool smallmem=config["smallmem"].specified;
  const bool discard_unks=config["discard_unks"].specified;
  const bool longint=config["longint"].specified;
  const bool average=config["average"].specified;
  const std::string readprob(config["readprob"].get_str());
  const std::string countsout(config["write_counts"].get_str());
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
    fprintf(stderr,"counts2kn might need to scan the input several times.  Cannot read the main data from stdin \"-\". Exit.\n");
    exit(-1);
  }
  TreeGram lm;

  std::string ss_sym;
  if (config["clear_history"].specified) {
    fprintf(stderr,"clearing history\n");
    ss_sym="<s>";
  }

  try {
    /* Construct the base kn-smoothed model */
    fprintf(stderr,"Estimating counts\n");
    ProbsAbs *abs=NULL;
    bool init_disc=true;

    /* Parse the arguments, create the right kind of model*/
    if (!smallvocab) {
      abs=new ProbsAbs_int_disc<int, double>(dataname, "", n, ss_sym, readprob, average, hashs);
    } else {
      abs=new ProbsAbs_int_disc<unsigned short, double>(dataname, "", n, ss_sym, readprob, average, hashs);
    } 
    
    fprintf(stderr, "The model will use ");
    if (smallvocab) fprintf(stderr,"small vocabulary (<65534) and ");
    else fprintf(stderr,"large vocabulary and ");
    fprintf(stderr,"absolute discounting.\n");

    
    //abs->cutoffs=cutoffs;
    //abs->discard_cutoffs=discard_cutoffs;
    //abs->discard_ngrams_with_unk=discard_unks;
    //kn->create_model((float) 0.0);
    
    if (vocabout.size()) {
      io::Stream out(vocabout, "w");
      abs->vocab.write(out.file);
    }

    if (countsout.size()) {
      io::Stream out(countsout, "w");
      abs->probs2ascii(out.file);
    }
    
  }
  catch (std::exception &e) {
    fprintf(stderr,"%s\n",e.what());
    exit(1); 
  }
}
