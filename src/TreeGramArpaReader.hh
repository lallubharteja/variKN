// Copyright (C) 2007  Vesa Siivola. 
// See licence.txt for the terms of distribution.

// Routines for reading and writing arpa format files from and to the 
// internal prefix tree format.
#ifndef TREEGRAMARPAREADER_HH
#define TREEGRAMARPAREADER_HH

#include <stdio.h>
#include "TreeGram.hh"

class TreeGramArpaReader {
public:
  void read(FILE *file, TreeGram *tree_gram);
  void write(FILE *file, TreeGram *tree_gram);
  void write_interpolated(FILE *file, TreeGram *treegram);
};

#endif /* TREEGRAMARPAREADER_HH */
