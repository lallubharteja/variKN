// Program to grow an n-gram model
#include "VarigramProbsFuncs.hh"
#include "conf.hh"
#include "io.hh"

int main(int argc, char **argv) {
  conf::Config config;
  config("Usage: varigram_probs [OPTIONS] textin LM_out\nProduces variable span n-gram from given text\n")
    ('o', "opti=FILE","arg","","The devel set for optimizing discount parameters. If not set, use leave-one-out discount estimates.")
    ('n', "norder=INT","arg","0","Maximal order included in the model (default unrestricted)")
    ('D',"dscale=FLOAT","arg","-1.0","Model size scale factor")
    ('E',"dscale2=FLOAT","arg","0","Model size scaling during pruning step (default no pruning=0)")
    ('a', "arpa","","","Output arpa instead of binary LM")
    ('B', "vocabin=FILE", "arg", "", "Restrict the vocabulary to given words")
    
    ('H', "hashsize=INT", "arg", "0", "Size of the reserved hash table. Speed vs. memory consumption.")
    ('s', "smallvocab", "", "", "Vocabulary is less than 65000 entries. Saves some memory.")
    ('C', "clear_history", "", "", "Clear LM history on each start of sentence tag (<s>).")
    ('N', "discard_unks", "", "", "Remove n-grams containing OOV words.")
    ('L', "longint", "", "", "Store counts in a long int type. Needed for big training sets.")
    ('V',"numngramstarget=INT","arg","0","Scale model down until there are less than V*1.03 ngrams in the model")
    ('F',"forcedisc=FLOAT","arg","-1.0", "Set all discounts to the given value.")
    ('P', "readprob=FILE","arg","", "Read the specified file that has probability associated with each word.\nEach line of the textprob has a word and an associated probability separated by space.");
  
  config.parse(argc,argv,2);
  
  const int max_order=config["norder"].get_int();
  const indextype hashs=config["hashsize"].get_int();
  const float dscale=std::max(0.00001, config["dscale"].get_double());
  const float dscale2=config["dscale2"].get_double();
  const int ngram_prune_target=config["numngramstarget"].get_int();
  const bool smallvocab=config["smallvocab"].specified;
  const int iter=1; //config["iter"].get_int();
  const bool discard_unks=config["discard_unks"].specified;
  const bool longint=config["longint"].specified;
  const float force_disc=config["forcedisc"].get_double();
  const std::string readprob(config["readprob"].get_str());

  std::string vocabname;
  if (config["vocabin"].specified) vocabname=config["vocabin"].get_str();

  std::string infilename(config.arguments.at(0));
  io::Stream::verbose=true;
  if (infilename=="-") {
    fprintf(stderr,"varigram_kn might need to scan the input several times.  Cannot read the main data from stdin \"-\". Exit.\n");
    exit(-1);
  }
  io::Stream out(config.arguments.at(1),"w");
  io::Stream::verbose=false;

  VarigramProbs *vg;
  if (!smallvocab)
    vg=new VarigramProbs_t<int, float>();
  else
    vg=new VarigramProbs_t<unsigned short, float>();
  
  if (dscale>0.0) vg->set_datacost_scale(dscale);
  if (dscale2>0.0) vg->set_datacost_scale2(dscale2); // use also pruning
  if (ngram_prune_target > 0) vg->set_ngram_prune_target(ngram_prune_target);
  if (max_order) vg->set_max_order(max_order);

  try {
    if (config["clear_history"].specified)
      vg->initialize(infilename, hashs, 
		     "<s>", readprob, vocabname);
    else 
      vg->initialize(infilename, hashs, 
		     "", readprob, vocabname);

    vg->set_discard_unks(discard_unks);
    vg->grow(iter);

    vg->write(out.file);
    out.close();
    delete vg;
  }
  catch (std::exception &e) {
    fprintf(stderr,"%s\n",e.what());
    exit(1); 
  }
}
