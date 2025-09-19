# import tables
import pickle
import numpy as np
import torch
from transformers import AutoProcessor, ClapModel
import logging
logger = logging.getLogger("SemanticSentenceModel")
from DataSequence import DataSequence


class SemanticSentenceModel(object):
    """This class defines a semantic sentence-level model

    """
    def __init__(self, model_id="laion/clap-htsat-unfused", device="cpu"):
        """Initializes a SemanticModel with the given [data] and [vocab].
        """
        self.device = device
        self.model_id = model_id
        self.semantic_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
        self.processor  = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    
    def get_ndim(self):
        """Returns the number of dimensions in this model.
        """

        return self.semantic_model.config.hidden_size
    ndim = property(get_ndim)

    def get_vindex(self):
        pass

    def __getitem__(self, word):
        """Returns the vector corresponding to the given [word].
        """

        with torch.no_grad():
            input_ids = self.processor(word, return_tensors="pt",padding=True).input_ids.to(self.device)
            output = self.semantic_model.get_text_features(input_ids).cpu()

        return output
    

    def encode_sentence(self,sentence):
        return self.__getitem__(sentence)


    def encode_story(self, story, return_ds=True, n_delays = 4):
        embeddings = []
        sentences=[]
        encoded_sentences = []
        for i,sentence in enumerate(story.chunks()):
            sentence = " ".join(sentence)
            sentences.append(sentence)

            with torch.no_grad():
                actual_sentence = " ".join(sentences[i-n_delays:i])
                encoded_sentences.append(actual_sentence)
                input_ids = self.processor(actual_sentence, return_tensors="pt").input_ids.to(self.device)
                output = self.semantic_model.get_text_features(input_ids).cpu()
            embeddings.append(output)   
        if return_ds:
            return DataSequence(torch.cat(embeddings, dim=0).numpy(), story.split_inds, story.data_times, story.tr_times) , sentences
        else:
            return torch.cat(embeddings, dim=0), encoded_sentences

    

    # def project_stims(self, stimwords):
    #     """Projects the stimuli given in [stimwords], which should be a list of lists
    #     of words, into this feature space. Returns the average feature vector across
    #     all the words in each stimulus.
    #     """
    #     logger.debug("Projecting stimuli..")
    #     stimlen = len(stimwords)
    #     ndim = self.data.shape[0]
    #     pstim = np.zeros((stimlen, ndim))
    #     vset = set(self.vocab)
    #     for t in range(stimlen):
    #         dropped = 0
    #         for w in stimwords[t]:
    #             dropped = 0
    #             if w in vset:
    #                 pstim[t] += self[w]
    #             else:
    #                 dropped += 1
            
    #         pstim[t] /= (len(stimwords[t])-dropped)

    #     return pstim

    # def uniformize(self):
    #     """Uniformizes each feature.
    #     """
    #     logger.debug("Uniformizing features..")
    #     R = np.zeros_like(self.data).astype(np.uint32)
    #     for ri in range(self.data.shape[0]):
    #         R[ri] = np.argsort(np.argsort(self.data[ri]))
        
    #     self.data = R.astype(np.float64)
    #     logger.debug("Done uniformizing...")

    # def gaussianize(self):
    #     """Gaussianizes each feature.
    #     """
    #     logger.debug("Gaussianizing features..")
    #     self.data = gaussianize_mat(self.data.T).T
    #     logger.debug("Done gaussianizing..")

    # def zscore(self, axis=0):
    #     """Z-scores either each feature (if axis is 0) or each word (if axis is 1).
    #     If axis is None nothing will be Z-scored.
    #     """
    #     if axis is None:
    #         logger.debug("Not Z-scoring..")
    #         return
        
    #     logger.debug("Z-scoring on axis %d"%axis)
    #     if axis==1:
    #         self.data = zscore(self.data.T).T
    #     elif axis==0:
    #         self.data = zscore(self.data)
    
    # def rectify(self):
    #     """Rectifies the features.
    #     """
    #     self.data = np.vstack([-np.clip(self.data, -np.inf, 0), np.clip(self.data, 0, np.inf)])
    
    # def clip(self, sds):
    #     """Clips feature values more than [sds] standard deviations away from the mean
    #     to that value.  Another method for dealing with outliers.
    #     """
    #     logger.debug("Truncating features to %d SDs.."%sds)
    #     fsds = self.data.std(1)
    #     fms = self.data.mean(1)
    #     newdata = np.zeros(self.data.shape)
    #     for fi in range(self.data.shape[0]):
    #         newdata[fi] = np.clip(self.data[fi],
    #                               fms[fi]-sds*fsds[fi],
    #                               fms[fi]+sds*fsds[fi])

    #     self.data = newdata
    #     logger.debug("Done truncating..")

    # def find_words_like_word(word,vocab_embeds=vocab_embeds,vocab=vocab, k=10):
    #     #encode word
    #     with torch.no_grad():
    #         input_ids = processor(word, return_tensors="pt").input_ids.to(device)
    #         output = semantic_model.get_text_features(input_ids).cpu()


    #     #compute similaritt

    #     similarities = vocab_embeds @ output.squeeze()

    #     #find top k
    #     topk = torch.topk(similarities, k=k, dim=0)
        
    #     #return sim, word
    #     return [(vocab[i], similarities[i].item()) for i in topk.indices]

    # def find_words_like_vec(self, vec, n=10, corr=True):
    #     """Finds the [n] words most like the given [vector].
    #     """
    #     nwords = len(self.vocab)
    #     if corr:
    #         corrs = np.nan_to_num([np.corrcoef(vec, self.data[:,wi])[1,0] for wi in range(nwords)])
    #         scorrs = np.argsort(corrs)
    #         words = list(reversed([(corrs[i], self.vocab[i]) for i in scorrs[-n:]]))
    #     else:
    #         proj = np.nan_to_num(np.dot(vec, self.data))
    #         sproj = np.argsort(proj)
    #         words = list(reversed([(proj[i], self.vocab[i]) for i in sproj[-n:]]))
    #     return words

    # def find_words_like_vecs(self, vecs, n=10, corr=True, distance_cull=None):
    #     """Find the `n` words most like each vector in `vecs`.
    #     """
    #     if corr:
    #         from text.npp import xcorr
    #         vproj = xcorr(vecs, self.data.T)
    #     else:
    #         vproj = np.dot(vecs, self.data)

    #     return np.vstack([self._get_best_words(vp, n, distance_cull) for vp in vproj])

    # def _get_best_words(self, proj, n=10, distance_cull=None):
    #     """Find the `n` words corresponding to the highest values in the vector `proj`.
    #     If `distance_cull` is an int, greedily find words with the following algorithm:
    #     1. Initialize the possible set of words with all words.
    #     2. Add the best possible word, w*. Remove w* from the possible set.
    #     3. Remove the `distance_cull` closest neighbors of w* from the possible set.
    #     4. Goto 2.
    #     """
    #     vocarr = np.array(self.vocab)
    #     if distance_cull is None:
    #         return vocarr[np.argsort(proj)[-n:][::-1]]
    #     elif not isinstance(distance_cull, int):
    #         raise TypeError("distance_cull should be an integer value, not %s" % str(distance_cull))

    #     poss_set = set(self.vocab)
    #     poss_set = np.arange(len(self.vocab))
    #     best_words = []
    #     while len(best_words) < n:
    #         # Find best word in poss_set
    #         best_poss = poss_set[proj[poss_set].argmax()]
    #         # Add word to best_words
    #         best_words.append(self.vocab[best_poss])
    #         # Remove nearby words (by L2-norm..?)
    #         bwdists = ((self.data.T - self.data[:,best_poss])**2).sum(1)
    #         nearest_inds = np.argsort(bwdists)[:distance_cull+1]
    #         poss_set = np.setdiff1d(poss_set, nearest_inds)

    #     return np.array(best_words)
    
    # def similarity(self, word1, word2):
    #     """Returns the correlation between the vectors for [word1] and [word2].
    #     """
    #     return np.corrcoef(self.data[:,self.vocab.index(word1)], self.data[:,self.vocab.index(word2)])[0,1]

    # def print_best_worst(self, ii, n=10):
    #     vector = self.data[ii]
    #     sv = np.argsort(self.data[ii])
    #     print ("Best:")
    #     print ("-------------")
    #     for ni in range(1,n+1):
    #         print ("%s: %0.08f"%(np.array(self.vocab)[sv[-ni]], vector[sv[-ni]]))
            
    #     print ("\nWorst:")
    #     print ("-------------")
    #     for ni in range(n):
    #         print ("%s: %0.08f"%(np.array(self.vocab)[sv[ni]], vector[sv[ni]]))
            
    #     print ("\n")



