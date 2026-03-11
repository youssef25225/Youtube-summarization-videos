from youtube_transcript_api import YouTubeTranscriptApi

def transcribe(url: str) -> str:
    video_id = _extract_video_id(url)

    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)

    return " ".join([t.text for t in transcript])


def _extract_video_id(url: str) -> str:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    elif "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    else:
        raise ValueError("Could not extract video ID from URL. Please use a standard YouTube link.")
