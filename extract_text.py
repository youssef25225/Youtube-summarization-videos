import yt_dlp
import whisper
import tempfile
import os


def download_audio_temp(url: str) -> str:
    """
    Download audio temporarily and return file path.
    File will exist only in system temp folder.
    """

    temp_dir = tempfile.mkdtemp()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    return filename


def transcribe(url: str) -> str:
    """
    Download audio temporarily → transcribe → delete file
    """

    file_path = download_audio_temp(url)

    model = whisper.load_model("base")

    result = model.transcribe(file_path)

    # delete file after transcription
    if os.path.exists(file_path):
        os.remove(file_path)

    return result["text"]
