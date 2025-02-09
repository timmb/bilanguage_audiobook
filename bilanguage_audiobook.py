#!/home/daniel/docs/code/python/bilanguage_audiobook/venv/bin/python


# Python
import sys
import subprocess
import os
import os.path
import subprocess
import re
import pprint
import time

#
import numpy as np


def secondsToVttTimestamp(i_seconds):
    """
    Params:
     i_seconds:
      (float)

    Returns:
     (str)
    """
    return "{:02d}".format(int(i_seconds / 60 / 60)) + \
        ":" + "{:02d}".format(int(i_seconds / 60) % 60) + \
        ":" + "{:02d}".format(int(i_seconds) % 60) + \
        "." + "{:03d}".format(int(i_seconds * 1000) % 1000)

def audio_float_to_int16(i_samples: np.ndarray, max_wav_value: float = 32767.0) -> np.ndarray:
    """Normalize audio and convert to int16 range"""
    rv = i_samples * (max_wav_value / max(0.01, np.max(np.abs(i_samples))))
    rv = np.clip(rv, -max_wav_value, max_wav_value)
    rv = rv.astype("int16")
    return rv


if __name__ == "__main__":
    # + Parse command line {{{

    COMMAND_NAME = "bilanguage_audiobook"

    def printUsage(i_outputStream):
        i_outputStream.write('''\
''' + COMMAND_NAME + '''
Transcribe, translate and reproduce spoken audio for the aid of language learning.

Usage:
======
''' + COMMAND_NAME + ''' <Media location> [Options...]

Params:
-------
 Media location:
  Location of some audio to reproduce.
  The location can be a file path, which will be accessed directly,
  or a URL, which will be downloaded into the temporary directory by shelling out to yt-dlp.
  The media itself can be an audio file, or a video, in which case its audio will be extracted.

Options:
--------

 Input
 -----
  -il/--input-language <code>
   Language of input audio, as a 2 letter code.
   eg.
    en
    ro
    es
    pt
   Default is 'ro'.

  -r/--range <start time in seconds>,<end time in seconds>
   Instead of processing the audio input in its entirety, specify a time range to work on.
    eg.
     60,120
      Include the range from 1 minute to 2 minutes.
   If either start time or end time is empty, it defaults to the start or end of the file.
    eg.
     60,
      Include the range from 1 minute to the end of the file.
     ,60
      Include the range from the beginning of the file to 1 minute.
   Pass this more than once to include multiple ranges.

 Transcription (with faster-whisper)
 -----------------------------------
  -m/--whisper-model <name>
   Which faster-whisper model to use to transcribe the input audio.
   eg.
    base
    large
    large-v2
    large-v3
    distil-large-v3
   Default is 'base'.

 Grouping
 --------
  -g/--group-by-sentence
   Attempt to regroup the words of transcription output,
   starting a new group with every sentence (ie. start a new one after each . or ? or !).

  --max-group-length <character count>
   Attempt to regroup the words of transcription output,
   allowing no more than this many characters total in each group.
   Can be used in conjunction with --group-by-sentence.

  If none of the above grouping options are used, faster-whisper's original output grouping is retained.

 Translation (with Argos Translate)
 ----------------------------------
  -gl/--generated-language <code>
   Language to translate to, as a 2 letter code.
   eg.
    en
    ro
    es
    pt
   Default is 'en'.

 Voice synthesis (with piper)
 ----------------------------
  -vn/--voice-name <name>
   Name of Piper ONNX voice to fetch from Hugging Face and use for converting English text to speech.
   See https://huggingface.co/rhasspy/piper-voices and the voices.json file therein.
   See also https://rhasspy.github.io/piper-samples for samples.
   The default is "en_GB-northern_english_male-medium".

  --voice-speaker-no <integer>
   If the selected voice contains multiple speakers, choose which speaker to use,
   starting at 0 and counting upwards.
   By default, use speaker 0.

 Outputs
 -------
  -ob/--output-base <path>
   Base path for output file(s) (without file extension).
   The output audio will be written to this path + "_bilanguage.mp3",
   and any supplementary files will follow a similar pattern.
   eg.
    "~/audio/MyStory"
     will create the file
      "~/audio/MyStory_bilanguage.mp3"
     and may also create
      "~/audio/MyStory_bilanguage.vtt"
   If the media location is a file path, this defaults to the same path without the original extension.
   If the media location is a URL, this option becomes mandatory.

  -s/--subtitles
   Write subtitles in VTT format to a file at the output base path + ".vtt".

 Other
 -----
  --help
   Show this help.
''')

    # Parameters and options, with their default values
    param_mediaLocation = None
    option_mediaLanguageCode = "ro"
    option_ranges = []
    option_whisperModelName = "base"
    #option_whisperExtraArgs
    option_groupBySentence = False
    option_groupMaxCharCount = None
    option_generatedLanguageCode = "en"
    option_piperVoiceName = "en_GB-northern_english_male-medium"
    option_piperVoiceSpeakerNo = None
    #option_gapLength
    option_outputBasePath = None
    option_subtitles = False

    # For each argument
    import sys
    argNo = 1
    while argNo < len(sys.argv):
        arg = sys.argv[argNo]
        argNo += 1

        # If it's an option
        if arg[0] == "-":
            if arg == "--help":
                printUsage(sys.stdout)
                sys.exit(0)

            elif arg == "-il" or arg == "--input-language":
                option_mediaLanguageCode = sys.argv[argNo]
                argNo += 1

            elif arg == "-r" or arg == "--range":
                if not re.match(r"[0-9.]*,[0-9.]*", sys.argv[argNo]):
                    print("ERROR: Value for -r/--range must be one or two numbers separated by a comma")
                    sys.exit(-1)
                #option_ranges.append(sys.argv[argNo].split(",", 1))
                start, end = sys.argv[argNo].split(",", 1)
                option_ranges.append([
                    None  if start == "" else  float(start),
                    None  if end == "" else  float(end)
                ])
                argNo += 1

            elif arg == "-m" or arg == "--whisper-model":
                option_whisperModelName = sys.argv[argNo]
                argNo += 1

            elif arg == "-g" or arg == "--group-by-sentence":
                option_groupBySentence = True

            elif arg == "--max-group-length":
                if not sys.argv[argNo].isnumeric():
                    print("ERROR: Value for --max-group-length must be an integer")
                    sys.exit(-1)
                option_groupMaxCharCount = int(sys.argv[argNo])
                argNo += 1

            elif arg == "-gl" or arg == "--generated-language":
                option_generatedLanguageCode = sys.argv[argNo]
                argNo += 1

            elif arg == "-vn" or arg == "--voice-name":
                option_piperVoiceName = sys.argv[argNo]
                argNo += 1

            elif arg == "-vn" or arg == "--voice-speaker-no":
                if not sys.argv[argNo].isnumeric():
                    print("ERROR: Value for --voice-speaker-no must be an integer")
                    sys.exit(-1)
                option_piperVoiceSpeakerNo = int(sys.argv[argNo])
                argNo += 1

            elif arg == "-ob" or arg == "--output-base":
                option_outputBasePath = sys.argv[argNo]
                argNo += 1

            elif arg == "-s" or arg == "--subtitles":
                option_subtitles = True

            else:
                print("ERROR: Unrecognised option: " + arg)
                print("(Run with --help to show command usage.)")
                sys.exit(-1)

        # Else if it's an argument
        else:
            if param_mediaLocation == None:
                param_mediaLocation = arg
            else:
                print("ERROR: Too many arguments.")
                print("(Run with --help to show command usage.)")
                sys.exit(-1)

    if param_mediaLocation == None:
        print("ERROR: Insufficient arguments.")
        print("(Run with --help to show command usage.)")
        sys.exit(-1)

    if option_outputBasePath == None:
        if param_mediaLocation.startswith("http"):
            print("ERROR: If media location is a URL then --output-file is required.")
            print("(Run with --help to show command usage.)")
            sys.exit(-1)
        else:
            option_outputBasePath = os.path.splitext(param_mediaLocation)[0]

    # + }}}

    # If media location is a URL then download the audio
    if param_mediaLocation.startswith("http"):
        print("Downloading audio with yt-dlp...")

        import tempfile
        tempWavFilePath = tempfile.gettempdir() + os.sep + "whisper_audio.wav"

        executableAndArguments = ["yt-dlp", param_mediaLocation, "--extract-audio", "--audio-format=wav", "-o", tempWavFilePath, "--force-overwrites"]
        print("Running: " + subprocess.list2cmdline(executableAndArguments))

        # Start subprocess
        import subprocess
        popen = subprocess.Popen(executableAndArguments, shell=False)
        # Wait for finish
        stdOutput, errOutput = popen.communicate()
        if popen.returncode != 0:
            print("ERROR: yt-dlp failed to download audio from URL.")
            sys.exit(popen.returncode)

        # Change param to use the file in the temp dir instead
        # TODO: This should be tempWavFilePath
        param_mediaLocation = "/tmp/whisper_audio.wav"

    """
    param_mediaLocation = "/mnt/ve/music/Radio/spanish/Hoy Hablamos/1716. Negar algo vs Negarse a algo.mp3"
    option_mediaLanguageCode = "es"

    param_mediaLocation = "/mnt/ve/music/Radio/Oscar_Wilde-Privighetoarea_si_trandafirul.mp3"
    param_mediaLocation = "/tmp/rom.mp3"
    option_mediaLanguageCode = "ro"

    option_ranges = []
    option_ranges = [[0, 200]]

    option_whisperModelName = "base"
    option_whisperModelName = "large-v3"

    option_groupBySentence = False
    option_groupMaxCharCount = None

    option_generatedLanguageCode = "en"

    option_piperVoiceName = "en_GB-northern_english_male-medium"
    option_piperVoiceSpeakerNo = None

    option_outputBasePath = "/tmp/out"
    option_subtitles = True
    """

    # + Transcribe with faster-whisper {{{

    print("=== Transcribing with faster-whisper...")

    # Construct model
    print("Constructing model: " + option_whisperModelName)
    import faster_whisper
    #model = faster_whisper.WhisperModel(option_whisperModelName, device="cuda", compute_type="float16")  # GPU with FP16
    #model = faster_whisper.WhisperModel(option_whisperModelName, device="cuda", compute_type="int8_float16")  # GPU with INT8
    model = faster_whisper.WhisperModel(option_whisperModelName, device="cpu", compute_type="int8")  # CPU with INT8

    #
    print("Transcribing...")
    transcribeKwargs = {}
    transcribeKwargs["audio"] = param_mediaLocation
    transcribeKwargs["language"] = option_mediaLanguageCode
    #transcribeKwargs["beam_size"] = 8
    #command += ["--max-context", "10"]
    #transcribeKwargs["best_of"] = 5
    transcribeKwargs["word_timestamps"] = True

    if len(option_ranges) > 0 and "," not in option_ranges:
        transcribeKwargs["clip_timestamps"] = []

        # Combine/include ranges which have only an end
        partRanges = [r  for r in option_ranges  if r[0] is None and r[1] is not None]
        if len(partRanges) > 0:
            transcribeKwargs["clip_timestamps"] += [0, max([partRange[1]  for partRange in partRanges])]

        # Include ranges which have both a start and end
        fullRanges = [r  for r in option_ranges  if r[0] is not None and r[1] is not None]
        for fullRange in fullRanges:
            transcribeKwargs["clip_timestamps"] += [float(fullRange[0]), float(fullRange[1])]

        # Combine/include ranges which have only a start
        partRanges = [r  for r in option_ranges  if r[0] is not None and r[1] is None]
        if len(partRanges) > 0:
            transcribeKwargs["clip_timestamps"] += [min([partRange[0]  for partRange in partRanges])]

    transcriptionSegments, transcriptionInfo = model.transcribe(**transcribeKwargs)
    #print(transcriptionInfo)
    segments = []
    for transcriptionSegment in transcriptionSegments:
        print("(" + "{:.2f}".format(transcriptionSegment.end / transcriptionInfo.duration * 100) + "%) " + transcriptionSegment.text)
        segments.append(transcriptionSegment)

    # + }}}

    # + Regroup {{{

    print("=== Grouping...")

    def regroupWords(i_words, i_breakSentences, i_maxLineLength):
        """
        Params:
         i_segments:
          (list of faster_whisper.transcribe.Word)
         i_breakSentences:
          (bool)
         i_maxLineLength:
          Either (int)
          or (None)

        Returns:
         (list of list of faster_whisper.transcribe.Word)
        """
        groups = []

        group = []
        groupLength = 0
        for word in i_words:
            # Add word to current group
            # and account for it's length
            group.append(word)
            groupLength += len(word.word)
            # If the current group is now long enough or ends a sentence,
            # finish it
            if (i_maxLineLength != None and groupLength >= i_maxLineLength) or \
               (i_breakSentences and (word.word.endswith(".") or word.word.endswith("?") or word.word.endswith("!"))):
                groups.append(group)
                group = []
                groupLength = 0
        # Finish final group
        if len(group) > 0:
            groups.append(group)
            group = []
            groupLength = 0

        return groups

    def renderGroupedWords(i_groupedWords, i_showTimestamps):
        """
        Params:
         i_groupedWords:
          (list of list of faster_whisper.transcribe.Word)
          As returned from the regroupWords() function.
         i_showTimestamps:
          (bool)
        """
        rv = ""

        for group in i_groupedWords:
            # Maybe render timestamp spanning whole group
            if i_showTimestamps:
                rv += secondsToVttTimestamp(group[0].start) + " --> " + secondsToVttTimestamp(group[-1].end) + "   "

            # Render words of group
            for word in group:
                rv += word.word
            rv += "\n"

        return rv

    groups = [segment.words  for segment in segments]
    print("in: " + str(len(segments)) + " groups")
    if option_groupBySentence or option_groupMaxCharCount != None:
        print("=== Regrouping sentences...")
        import itertools
        groups = regroupWords(list(itertools.chain.from_iterable(groups)), option_groupBySentence, option_groupMaxCharCount)
    print("out: " + str(len(groups)) + " groups")
    #print(renderGroupedWords(groups, True))

    # + }}}

    # + Translate with Argos Translate {{{

    print("=== Translating with Argos Translate...")

    # Choose language pair package, and download it if don't have it already
    import argostranslate.package
    argostranslate.package.update_package_index()
    availablePackages = argostranslate.package.get_available_packages()
    selectedPackage = next(filter(lambda x: x.from_code == option_mediaLanguageCode and x.to_code == option_generatedLanguageCode, availablePackages))
    packageDownloadPath = selectedPackage.download()
    argostranslate.package.install_from_path(packageDownloadPath)

    # Translate all groups
    import argostranslate.translate
    translations = []
    for groupNo, group in enumerate(groups):
        sourceLanguageText = "".join([word.word  for word in group])
        generatedLanguageText = argostranslate.translate.translate(sourceLanguageText, option_mediaLanguageCode, option_generatedLanguageCode)
        translations.append(generatedLanguageText)
        print("(" + "{:.2f}".format((groupNo+1) / len(groups) * 100) + "%) " + sourceLanguageText.strip())
        print("          > " + generatedLanguageText.strip())

    # + }}}

    # + Initialize subtitle file {{{

    # If wanted by options
    if option_subtitles:
        vttFile = open(option_outputBasePath + "_bilanguage.vtt", "w")
        vttFile.write("WEBVTT\n\n")

        note = "Command line: " + subprocess.list2cmdline(sys.argv) + "\n"
        note += "transcribeKwargs: " + str(transcribeKwargs) + "\n"
        vttFile.write("NOTE\n")
        vttFile.write(note)
        vttFile.write("\n")

    # + }}}

     # + Initialize text-to-speech (OpenAI) {{{

    print("=== Initializing OpenAI TTS...")
    from openai import OpenAI
    from pathlib import Path
    import tempfile
    import pydub

    # beep = pydub.generators.Sawtooth(440, duty_cycle = 0.2).to_audio_segment(duration = 100) - 10.0  # quieten by 10 dB
    gap = pydub.AudioSegment.silent(duration = 600 * 0)  # milliseconds


    # Initialize OpenAI client
    openai = OpenAI()

    # + }}}

    # + Synthesize audio {{{

    print("=== Synthesizing audio...")

    source = pydub.AudioSegment.from_file(param_mediaLocation)
    sourcePosInSeconds = 0.0

    outAudioSegments = []
    outPosInSeconds = 0.0

    for groupNo, group in enumerate(groups):
        if groupNo + 1 < len(groups):
            nextGroup = groups[groupNo + 1]
            endSourcePosInSeconds = (group[-1].end + nextGroup[0].start) / 2
        else:
            endSourcePosInSeconds = group[-1].end

        print(f"({(groupNo / len(groups) * 100):.2f}%) {secondsToVttTimestamp(sourcePosInSeconds)} .. {secondsToVttTimestamp(endSourcePosInSeconds)}")

        # Append source audio segment
        if option_subtitles:
            outStartPos = outPosInSeconds
        audioSegment = source[int(sourcePosInSeconds * 1000):int(endSourcePosInSeconds * 1000)]
        outAudioSegments.append(audioSegment)
        outPosInSeconds += audioSegment.duration_seconds
        if option_subtitles:
            vttFile.write(f"{secondsToVttTimestamp(outStartPos)} --> {secondsToVttTimestamp(outPosInSeconds)}\n")
            vttFile.write(" " + "".join(word.word for word in group) + "\n\n")
            vttFile.flush()

        outAudioSegments.append(gap)
        outPosInSeconds += gap.duration_seconds

        # Generate speech and append
        generatedLanguageText = translations[groupNo]
        temp_audio_path = Path(tempfile.gettempdir()) / f"speech_{groupNo}.mp3"

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="onyx",
            input=generatedLanguageText,
        ) as response:
            response.stream_to_file(temp_audio_path)

        generated_audio = pydub.AudioSegment.from_file(temp_audio_path, format="mp3") - 5.0
        outAudioSegments.append(generated_audio)
        outPosInSeconds += generated_audio.duration_seconds
        if option_subtitles:
            vttFile.write(f"{secondsToVttTimestamp(outStartPos)} --> {secondsToVttTimestamp(outPosInSeconds)}\n")
            vttFile.write(f" {generatedLanguageText}\n\n")
            vttFile.flush()

        outAudioSegments.append(gap)
        outPosInSeconds += gap.duration_seconds

        # Append source audio segment again
        if option_subtitles:
            outStartPos = outPosInSeconds
        audioSegment = source[int(sourcePosInSeconds * 1000):int(endSourcePosInSeconds * 1000)]
        outAudioSegments.append(audioSegment)
        outPosInSeconds += audioSegment.duration_seconds
        if option_subtitles:
            vttFile.write(f"{secondsToVttTimestamp(outStartPos)} --> {secondsToVttTimestamp(outPosInSeconds)}\n")
            vttFile.write(" " + "".join(word.word for word in group) + "\n\n")
            vttFile.flush()

        sourcePosInSeconds = endSourcePosInSeconds

    # + }}}

    # + Join audio {{{

    print("=== Joining audio...")

    def joinAudioSegments(i_audioSegments, i_frameRate, i_sampleByteCount, i_channelCount):
        """
        Convert a sequence of audio segments to the same format and concatenate them.

        Pydub doesn't have a built-in function for this. It overloads the '+' operator for concatenation,
        but after a large quantity of small concatenations it becomes prohibitively slow.
        See https://github.com/jiaaro/pydub/issues/256

        Params:
         i_audioSegments:
          (list of pydub.AudioSegment)
         i_frameRate:
          (int)
          Frame rate to resample everything to
          eg. 44100
         i_sampleByteCount:
          (int)
          Bytes per sample to convert everything to
          eg. 2 (for 16-bit samples)
         i_channelCount:
          (int)
          Channel count of output sound
          eg. 2 (stereo)

        Returns:
         (pydub.AudioSegment)
        """
        # Get all audio segments in desired output format
        i_audioSegments = [audioSegment.set_frame_rate(i_frameRate).set_sample_width(i_sampleByteCount).set_channels(i_channelCount)
                           for audioSegment in i_audioSegments]
    
        # Get total sample count of all segments
        frameCount = int(sum([audioSegment.frame_count()  for audioSegment in i_audioSegments]))
        sampleCount = frameCount * i_channelCount
    
        # Create a Python array.array at that size,
        # asking the first converted pydub segment for the appropriate type code it should use
        import array
        outputSamples = array.array(i_audioSegments[0].array_type, [0]*sampleCount)

        # For each audio segment,
        # get pydub to give us an array.array and copy the samples into the output array.array
        outputSampleNo = 0
        for audioSegmentNo, audioSegment in enumerate(i_audioSegments):
            if audioSegmentNo % 200 == 0:
                print("(" + "{:.2f}".format((audioSegmentNo) / len(i_audioSegments) * 100) + "%)")
            audioSegmentSamples = audioSegment.get_array_of_samples()
            for audioSegmentSampleNo in range(len(audioSegmentSamples)):
                outputSamples[outputSampleNo] = int(audioSegmentSamples[audioSegmentSampleNo])
                outputSampleNo += 1
    
        # Convert back to a pydub AudioSegment
        return i_audioSegments[0]._spawn(outputSamples)

    outAudioSegment = joinAudioSegments(outAudioSegments, source.frame_rate, source.sample_width, source.channels)

    # + }}}

    # Export audio to disk
    print("=== Exporting audio to " + option_outputBasePath + "_bilanguage.mp3")
    outAudioSegment.export(option_outputBasePath + "_bilanguage.mp3", format="mp3")

    # Cleanup
    if option_subtitles:
        print("=== Closing subtitles in " + option_outputBasePath + ".vtt")
        vttFile.close()
