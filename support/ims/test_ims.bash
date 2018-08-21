#!/bin/bash
if [ $# -lt 3 ]; then
  echo "$0 modelDir testFile savePath"
  exit
fi
modelDir=$1
testFile=$2
savePath=$3
gradle run -DmainClass=sg.edu.nus.comp.nlp.ims.implement.CTester --args="-prop jwnl-properties.xml -token 1 -pos 1 -split 1 -lemma 1 -r sg.edu.nus.comp.nlp.ims.io.CAllWordsResultWriter $testFile $modelDir $modelDir $savePath -f sg.edu.nus.comp.nlp.ims.feature.CAllWordsFeatureExtractorCombination"
