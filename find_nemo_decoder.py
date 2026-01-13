import nemo.collections.asr as nemo_asr
import inspect

# Find the decoder implementation
model_class = nemo_asr.models.EncDecRNNTBPEModel
print("Model file:", inspect.getfile(model_class))

# Try to find the decoding module
try:
    from nemo.collections.asr.parts.submodules import rnnt_greedy_decoding
    print("Greedy decoder:", inspect.getfile(rnnt_greedy_decoding))
except ImportError as e:
    print("Could not import rnnt_greedy_decoding:", e)

# Try beam search
try:
    from nemo.collections.asr.parts.submodules import rnnt_beam_decoding
    print("Beam decoder:", inspect.getfile(rnnt_beam_decoding))
except ImportError as e:
    print("Could not import rnnt_beam_decoding:", e)

# Find decoding module
try:
    import nemo.collections.asr.parts.submodules.rnnt_decoding as rnnt_decoding
    print("RNNT decoding module:", inspect.getfile(rnnt_decoding))
except ImportError as e:
    print("Could not import rnnt_decoding:", e)
