#!/bin/bash
if [ $# -lt 3 ]; then
  echo $0 train.xml train.key savedir
  exit
fi
cd "$(dirname "$0")"
gradle run -DmainClass=sg.edu.nus.comp.nlp.ims.implement.CTrainModel --args="-prop jwnl-properties.xml -token 1 -pos 1 -split 1 -lemma 1 $1 $2 $3 -f sg.edu.nus.comp.nlp.ims.feature.CFeatureExtractorCombination"
