import math
from typing import Tuple, List
import numpy as np

__all__ = ["Seq2SeqBeam", "TransformerBeam"]


class Seq2SeqBeam(object):
    """
    numpy version of vietocr's beam for seq2seq model
    """

    def __init__(
        self,
        batch_size: int = 1,
        beam_size: int = 5,
        start_token_id: int = 1,
        end_token_id: int = 2,
    ):
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id

        self.top_sentence_ended = np.zeros((batch_size,), dtype=bool)

        # because of encoder's outputs tiling, beam/vocab index's item have shape of [beam_size, batch_size]
        # [S1,S2,...,Sn] -> [S1,S2,...,Sn,S1,S2,...,Sn,...]
        # beam index
        self.prev_ks: List[np.ndarray] = []
        # vocab index
        self.next_ys: List[np.ndarray] = [
            np.full(
                (beam_size, batch_size),
                fill_value=start_token_id,
                dtype=np.longlong,  # onnx's int64 issue
            )
        ]  # remove padding

        # probability of decoded sequence is average probability of its characters
        # https://github.com/pbcquoc/vietocr/issues/43
        self.char_probs = np.zeros((beam_size, batch_size), dtype=np.float32)

        # log-prob scores
        self.current_scores = np.zeros((beam_size, batch_size), dtype=np.float32)

        # for each sent, finished holds list of tuple (score, prob, step, beam) if eos appears
        self.finished: List[List[Tuple[float, float, int, int]]] = [
            [] for _ in range(batch_size)
        ]

    def get_current_state(self) -> np.ndarray:
        # get last state [beam_size*batch_size,]
        return self.next_ys[-1].reshape((-1,))

    def advance(self, log_prob: np.ndarray):
        # log_prob: [beam_size*batch_size, vocab_size]
        log_prob = log_prob.reshape((self.beam_size, self.batch_size, -1))
        vocab_size = log_prob.shape[2]

        if len(self.prev_ks) > 0:
            # \sum{\log{xi}} = \log{x1*...*xn}
            beam_scores = (
                self.current_scores[:, :, None] + log_prob
            )  # [beam_size, batch_size, vocab_size]

            prob = np.exp(log_prob)
            prob[:, :, self.end_token_id] = 0  # ignore probability of "eos" token
            char_probs = self.char_probs[:, :, None] + prob

            # don't let EOS have children (by ignoring ended beams)
            for i, j in zip(
                *np.where(self.next_ys[-1] == self.end_token_id)
            ):  # i: beam, j: batch
                beam_scores[i, j] = -math.inf  # [vocab_size]

        else:  # in first step, get first beam only to avoid duplicating beams
            beam_scores = log_prob[0:1]  # [1, batch_size, vocab_size]

            char_probs = np.exp(log_prob[0:1])
            char_probs[:, :, self.end_token_id] = 0  # ignore probability of "eos" token

        beam_scores = beam_scores.transpose((0, 2, 1)).reshape(
            (-1, self.batch_size)
        )  # [beam_size*vocab_size, batch_size]

        char_probs = char_probs.transpose((0, 2, 1)).reshape(
            (-1, self.batch_size)
        )  # [beam_size*vocab_size, batch_size]

        # get top-k (descreasing score) beams
        top_score_ids = np.argsort(beam_scores, axis=0)[-self.beam_size :][
            ::-1
        ]  # [beam_size, batch_size]

        # get top-k beam index
        prev_k = top_score_ids // vocab_size
        self.prev_ks.append(prev_k)

        # get top-k vocab index
        next_y = top_score_ids % vocab_size
        self.next_ys.append(next_y)

        # compute current scores
        self.current_scores = np.stack(
            [beam_scores[top_score_ids[:, i], i] for i in range(self.batch_size)]
        ).T  # [beam_size, batch_size]

        self.char_probs = np.stack(
            [char_probs[top_score_ids[:, i], i] for i in range(self.batch_size)]
        ).T  # [beam_size, batch_size]

        # check ended (top beam (the highest score)'s vocab index is EOS)
        self.top_sentence_ended = np.logical_or(
            self.top_sentence_ended, next_y[0] == self.end_token_id
        )  # [batch_size,]

        # check if sent is ended, add current state to finished
        for i, j in zip(*np.where(next_y == self.end_token_id)):  # i: beam, j: batch
            self.finished[j].append(
                (
                    self.current_scores[i, j],
                    self.char_probs[i, j]
                    / (len(self.next_ys) - 2),  # ignore "sos", "eos" tokens
                    len(self.next_ys) - 1,
                    i,
                )
            )

    def done(self) -> bool:
        return np.all(self.top_sentence_ended)  # all sentences are ended

    def sort_finished(self, minimum: int = 1):
        # fill beams to finished for having at least minimum ended beams
        for i, fin in enumerate(self.finished):  # i: batch
            for j in range(self.beam_size):  # j: beam
                if len(fin) >= minimum:
                    break

                fin.append(
                    (
                        self.current_scores[j, i],
                        self.char_probs[j, i]
                        / (len(self.next_ys) - 2),  # ignore "sos", "eos" tokens
                        len(self.next_ys) - 1,
                        j,
                    )
                )

            # sort finished based on scores, top-ended at 0 index
            self.finished[i] = sorted(fin, key=lambda x: x[0], reverse=True)

    def get_hypothesises(
        self,
    ) -> Tuple[List[List[int]], List[float]]:  # get hypothesises for batch of sent
        translated_sentence = list()
        char_probs = list()
        for i, fin in enumerate(self.finished):
            hyp = list()
            _, prob, t, k = fin[0]  # get best candidate, t is EOS step
            for j in range(t, 0, -1):  # step j
                hyp.append(self.next_ys[j][k, i])  # k: beam, i: batch
                k = self.prev_ks[j - 1][k, i]

            translated_sentence.append([self.start_token_id] + hyp[::-1])
            char_probs.append(prob)

        return translated_sentence, char_probs


class TransformerBeam(Seq2SeqBeam):
    """
    numpy version of vietocr's beam for transformer model
    """

    def get_current_state(self) -> np.ndarray:
        # get all states [Ldec,beam_size*batch_size]
        return np.stack(self.next_ys).reshape((len(self.next_ys), -1))
