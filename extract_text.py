from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

def transcribe(url: str) -> str:
    """
    Fetch the transcript directly from YouTube.
    No audio download or Whisper needed.
    Raises a clear error if the video has no transcript.
    """
    video_id = _extract_video_id(url)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        raise ValueError("This video has disabled transcripts.")
    except NoTranscriptFound:
        raise ValueError("No transcript found for this video. Try a video with captions enabled.")

    return " ".join([t["text"] for t in transcript])


def _extract_video_id(url: str) -> str:
    """
    Extract video ID from various YouTube URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://youtube.com/watch?v=VIDEO_ID&other=params
    """
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    elif "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    else:
        raise ValueError("Could not extract video ID from URL. Please use a standard YouTube link.")
