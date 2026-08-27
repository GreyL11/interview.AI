from app.audio.base import AudioChannel, AudioError, DeviceInfo
from app.core.logging import get_logger

logger = get_logger(__name__)


def _sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        # OSError too: sounddevice raises it when the PortAudio DLL is missing,
        # which on Windows is a far more common failure than a missing package.
        raise AudioError(
            f"Audio capture is unavailable ({exc}). Install the audio extras and "
            "ensure PortAudio is present."
        ) from exc
    return sounddevice


def audio_available() -> bool:
    try:
        _sounddevice()
        return True
    except AudioError:
        return False


def list_devices() -> list[DeviceInfo]:
    """Enumerate capture devices, tagging which can hear the interviewer.

    On Windows the interviewer's voice arrives through WASAPI loopback, which
    PortAudio exposes as an *output* device that can be opened for input. Those
    are reported as LOOPBACK; ordinary inputs are reported as MIC.
    """
    sd = _sounddevice()
    devices: list[DeviceInfo] = []

    try:
        default_input = sd.default.device[0]
    except Exception:
        default_input = -1

    loopback_hostapis = {
        i for i, api in enumerate(sd.query_hostapis()) if "WASAPI" in api.get("name", "")
    }

    for index, device in enumerate(sd.query_devices()):
        max_in = device.get("max_input_channels", 0)
        max_out = device.get("max_output_channels", 0)
        hostapi = device.get("hostapi", -1)

        if max_in > 0:
            devices.append(DeviceInfo(
                index=index,
                name=device.get("name", f"device {index}"),
                channel=AudioChannel.MIC,
                sample_rate=int(device.get("default_samplerate", 16_000)),
                is_default=(index == default_input),
            ))
        elif max_out > 0 and hostapi in loopback_hostapis:
            devices.append(DeviceInfo(
                index=index,
                name=f"{device.get('name', f'device {index}')} (loopback)",
                channel=AudioChannel.LOOPBACK,
                sample_rate=int(device.get("default_samplerate", 16_000)),
                is_default=False,
            ))

    return devices


def default_device(channel: AudioChannel) -> DeviceInfo | None:
    candidates = [d for d in list_devices() if d.channel == channel]
    if not candidates:
        return None
    return next((d for d in candidates if d.is_default), candidates[0])
