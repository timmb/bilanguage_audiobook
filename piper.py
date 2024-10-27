# type: PiperConfig
#  (dict)
#  Keys:
#   phoneme_count:
#    (int)
#    Number of phonemes
#   speaker_count:
#    (int)
#    Number of speakers
#   audio_sampleRate:
#    (int)
#    Sample rate of output audio
#   espeak_voice:
#    (str)
#    Name of espeak-ng voice or alphabet
#    eg.
#     "ar"
#   inference_length_scale:
#    (float)
#   inference_noise_scale:
#    (float)
#   inference_noise_w:
#    (float)
#   phoneme_stringToIdMap:
#    (Mapping[str, Sequence[int]])
#    Phoneme -> [id,]
#   phoneme_type:
#    (str)
#    One of
#     "espeak"
#     "text"

# Python std
import json

def loadModelConfig(i_jsonFilePath):
    with open(i_jsonFilePath, "r", encoding="utf-8") as f:
        modelConfig = json.load(f)

    rv = {}

    rv["phoneme_count"] = modelConfig["num_symbols"]

    rv["phoneme_stringToIdMap"] = modelConfig["phoneme_id_map"]

    rv["phoneme_type"] = "espeak"
    if "phoneme_type" in modelConfig:
        rv["phoneme_type"] = modelConfig["phoneme_type"]

    rv["speaker_count"] = modelConfig["num_speakers"]

    rv["audio_sampleRate"] = modelConfig["audio"]["sample_rate"]

    rv["inference_lengthScale"] = 1.0
    if "inference" in modelConfig and "length_scale" in modelConfig["inference"]:
        rv["inference_lengthScale"] = modelConfig["inference"]["length_scale"]

    rv["inference_noiseScale"] = 0.667
    if "inference" in modelConfig and "noise_scale" in modelConfig["inference"]:
        rv["inference_noiseScale"] = modelConfig["inference"]["noise_scale"]

    rv["inference_noiseW"] = 0.8
    if "inference" in modelConfig and "noise_scale" in modelConfig["inference"]:
        rv["inference_noiseW"] = modelConfig["inference"]["noise_w"]

    rv["espeak_voice"] = modelConfig["espeak"]["voice"]

    return rv


# Python std
import json
import logging
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

#
import numpy as np
import onnxruntime

import piper_phonemize


#
_LOGGER = logging.getLogger(__name__)



class PiperVoice:
    """
    An ONNX model and config
    """
    # Member variables:
    #  modelConfig:
    #   (PiperConfig)
    #  inferenceSession:
    #   (onnxruntime.InferenceSession)

    def __init__(self,
                 i_modelPath: Union[str, pathlib.Path],
                 i_configPath: Optional[Union[str, pathlib.Path]] = None,
                 i_useCuda: bool = False):
        """
        Params:
         i_modelPath:
          (str or pathlib.Path)
          ONNX model file
          eg. "en_GB-northern_english_male-medium.onnx"
         i_configPath: Optional[Union[str:
          Either (str or pathlib.Path)
           Model's JSON config file
           eg. "en_GB-northern_english_male-medium.onnx.json"
          or (None)
           Default to the same as i_modelPath plus ".json".
         i_useCuda:
          (bool)
        """
        self.modelConfig = loadModelConfig(i_configPath)

        #providers: List[Union[str, Tuple[str, Dict[str, Any]]]]
        if i_useCuda:
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})]
        else:
            providers = ["CPUExecutionProvider"]
        self.inferenceSession = onnxruntime.InferenceSession(str(i_modelPath), sess_options = onnxruntime.SessionOptions(), providers = providers)

    def phonemize(self, i_text: str) -> List[List[str]]:
        """
        Given some text, split it to sentences and convert each sentence to phoneme strings, using a utility from piper_phonemize.

        Returns:
         (list)
         The sentences that were present in i_text.
         Each element is:
          (list)
          The phonemes of the sentence.
          Each element is:
           (str)
        """
        # If have configured espeak-type phonemes,
        # use phonemize_espeak()
        if self.modelConfig["phoneme_type"] == "espeak":
            # If an Arabic voice, do diacritization
            # https://github.com/mush42/libtashkeel/
            if self.modelConfig["espeak_voice"] == "ar":
                i_text = piper_phonemize.tashkeel_run(i_text)

            return piper_phonemize.phonemize_espeak(i_text, self.modelConfig["espeak_voice"])

        # Else if have configured text-type phonemes,
        # use phonemize_codepoints()
        elif self.modelConfig["phoneme_type"] == "text":
            return piper_phonemize.phonemize_codepoints(i_text)

        #
        else:
            raise ValueError("Unexpected phoneme type: " + self.modelConfig["phoneme_type"])


    PAD = "_"  # padding (0)
    BOS = "^"  # beginning of sentence
    EOS = "$"  # end of sentence

    def phonemes_to_ids(self, i_phonemeStrings: List[str]) -> List[int]:
        """
        Given a list of phoneme strings,
        look up their ID numbers in the current model's phoneme to id map.

        Params:
         i_phonemeStrings:
          (list of string)

        Returns:
         (list of int)
        """
        phonemeIdMap = self.modelConfig["phoneme_stringToIdMap"]

        phonemeIds = list(phonemeIdMap[PiperVoice.BOS])

        for phonemeString in i_phonemeStrings:
            if phonemeString not in phonemeIdMap:
                _LOGGER.warning("Phoneme string missing from model's id map: %s", phonemeString)
                continue

            phonemeIds.extend(phonemeIdMap[phonemeString])
            phonemeIds.extend(phonemeIdMap[PiperVoice.PAD])

        phonemeIds.extend(phonemeIdMap[PiperVoice.EOS])

        return phonemeIds

    def synthesize_fromPhonemeIds(self,
                                  i_phonemeIds: List[int],
                                  i_speakerId: Optional[int] = None,
                                  i_inference_lengthScale: Optional[float] = None,
                                  i_inference_noiseScale: Optional[float] = None,
                                  i_inference_noiseW: Optional[float] = None) -> np.ndarray:
        """
        Synthesize mono audio samples from phoneme ids.

        Params:
         i_phonemeIds:
          (list of int)
          eg. as returned from phonemes_to_ids()
         i_speakerId:
          Either (int)
           If the model supports multiple speakers,
           which speaker number (0-based) to use.
          or (None)
         i_inference_lengthScale:
          Either (float)
          or (None)
           Default to the value that is in the model's JSON config file.
         i_inference_noiseScale:
          Either (float)
          or (None)
           Default to the value that is in the model's JSON config file.
         i_inference_noiseW:
          Either (float)
          or (None)
           Default to the value that is in the model's JSON config file.

        Returns:
         (numpy.ndarray with dtype=float32)
        """

        # Apply default arguments, from model config
        if i_inference_lengthScale is None:
            i_inference_lengthScale = self.modelConfig["inference_lengthScale"]
        if i_inference_noiseScale is None:
            i_inference_noiseScale = self.modelConfig["inference_noiseScale"]
        if i_inference_noiseW is None:
            i_inference_noiseW = self.modelConfig["inference_noiseW"]

        #
        phonemeIds_array = np.expand_dims(np.array(i_phonemeIds, dtype=np.int64), 0)
        phonemeIds_lengths = np.array([phonemeIds_array.shape[1]], dtype=np.int64)
        scales = np.array([i_inference_noiseScale, i_inference_lengthScale, i_inference_noiseW], dtype=np.float32)

        args = {
            "input": phonemeIds_array,
            "input_lengths": phonemeIds_lengths,
            "scales": scales
        }

        if self.modelConfig["speaker_count"] <= 1:
            i_speakerId = None
        elif (self.modelConfig["speaker_count"] > 1) and (i_speakerId is None):
            i_speakerId = 0
        if i_speakerId is not None:
            args["sid"] = np.array([i_speakerId], dtype=np.int64)

        # Synthesize through Onnx
        return self.inferenceSession.run(None, args, )[0].squeeze((0, 1)).squeeze()

    def synthesize_fromText(self,
                            i_text: str,
                            i_speakerId: Optional[int] = None,
                            i_inference_lengthScale: Optional[float] = None,
                            i_inference_noiseScale: Optional[float] = None,
                            i_inference_noiseW: Optional[float] = None,
                            i_sentencePostgapInSeconds: float = 0.0) -> np.ndarray:
        """
        Synthesize mono audio from text.

        Params:
         i_text:
          (str)
         i_phonemeIds:
          (list of int)
          eg. as returned from phonemes_to_ids()
         i_speakerId, i_inference_lengthScale, i_inference_noiseScale, i_inference_noiseW:
          Same as in synthesize_fromPhonemeIds().
         i_sentencePostgapInSeconds:
          (float)
          Seconds of silence to append to each sentence.

        Returns:
         (numpy.ndarray with dtype=float32)
        """
        sentence_phonemes = self.phonemize(i_text)
        #print(sentence_phonemes)

        sentenceSilenceSampleCount = int(i_sentencePostgapInSeconds * self.modelConfig["audio_sampleRate"])

        samples = np.array([])
        for phonemes in sentence_phonemes:
            #print("inner: ", phonemes)
            phoneme_ids = self.phonemes_to_ids(phonemes)
            sentenceSamples = self.synthesize_fromPhonemeIds(phoneme_ids,
                                                             i_speakerId = i_speakerId,
                                                             i_inference_lengthScale = i_inference_lengthScale,
                                                             i_inference_noiseScale = i_inference_noiseScale,
                                                             i_inference_noiseW = i_inference_noiseW)
            # Append silence
            samples = np.concatenate([samples, sentenceSamples, np.zeros([sentenceSilenceSampleCount])])
        return samples

#import wave
#    def synthesize(self,
#                   i_text: str,
#                   wav_file: wave.Wave_write,
#                   speaker_id: Optional[int] = None,
#                   length_scale: Optional[float] = None,
#                   noise_scale: Optional[float] = None,
#                   noise_w: Optional[float] = None,
#                   sentence_silence: float = 0.0):
#        """Synthesize WAV audio from text."""
#        wav_file.setframerate(self.modelConfig["audio_sampleRate"])
#        wav_file.setsampwidth(2)  # 16-bit
#        wav_file.setnchannels(1)  # mono
#
#        for audio_bytes in self.synthesize_stream_raw(
#            i_text,
#            speaker_id = speaker_id,
#            length_scale = length_scale,
#            noise_scale = noise_scale,
#            noise_w = noise_w,
#            sentence_silence = sentence_silence,
#        ):
#            wav_file.writeframes(audio_bytes)

class RestartablePiperVoice:
    """
    The PiperVoice, or more specifically the underlying onnxruntime.InferenceSession,
    has a habit of crashing into a state where it refuses to synthesise anything else,
    only throwing up exceptions of the form onnxruntime.capi.onnxruntime_pybind11_state....

    It seems to happen if you ask it to synthesis the same sentence (more specifically,
    the same sequence of phonemes) twice in the same session. There may be other causes too.
    
    As a brute workaround, this class wraps a PiperVoice instance and if an exception occurs
    during synthesis it will simply recreate the instance and try again.
    """
    def __init__(self,
                 i_modelPath: Union[str, pathlib.Path],
                 i_configPath: Optional[Union[str, pathlib.Path]] = None,
                 i_useCuda: bool = False):
        self.modelPath = i_modelPath
        self.configPath = i_configPath
        self.useCuda = i_useCuda

        self.recreatePiperVoice()

    def recreatePiperVoice(self):
        self.piperVoice = PiperVoice(self.modelPath, self.configPath, self.useCuda)

    #self.phonemize = self.piperVoice.phonemize
    #self.phonemes_to_ids = self.piperVoice.phonemes_to_ids
    #self.synthesize_fromPhonemeIds = self.piperVoice.synthesize_fromPhonemeIds

    def synthesize_fromText(self,
                            i_text: str,
                            i_speakerId: Optional[int] = None,
                            i_inference_lengthScale: Optional[float] = None,
                            i_inference_noiseScale: Optional[float] = None,
                            i_inference_noiseW: Optional[float] = None,
                            i_sentencePostgapInSeconds: float = 0.0) -> np.ndarray:
        try:
            return self.piperVoice.synthesize_fromText(i_text, i_speakerId, i_inference_lengthScale, i_inference_noiseScale, i_inference_noiseW, i_sentencePostgapInSeconds)
        except (onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument, onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException) as e:
            self.recreatePiperVoice()
            return self.synthesize_fromText(i_text, i_speakerId, i_inference_lengthScale, i_inference_noiseScale, i_inference_noiseW, i_sentencePostgapInSeconds)
