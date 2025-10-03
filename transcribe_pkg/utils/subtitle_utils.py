#!/usr/bin/env python3
"""
Utility functions for generating subtitle formats from transcriptions with timestamps.

Supported formats:
- SRT (SubRip Text): The most common subtitle format
- VTT (Web Video Text Tracks): HTML5 standard for subtitles

Both formats include timestamp information for synchronizing text with audio/video.
"""

import os
import logging
import functools
from datetime import timedelta
from typing import Any

from transcribe_pkg.utils.logging_utils import get_logger

@functools.lru_cache(maxsize=1024)
def format_timestamp_srt(seconds: float) -> str:
    """
    Format seconds as SRT timestamp (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds (float)
        
    Returns:
        Formatted timestamp string in SRT format
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"


@functools.lru_cache(maxsize=1024)
def format_timestamp_vtt(seconds: float) -> str:
    """
    Format seconds as VTT timestamp (HH:MM:SS.mmm).
    
    Args:
        seconds: Time in seconds (float)
        
    Returns:
        Formatted timestamp string in VTT format
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"


def generate_srt(segments: list[dict[str, Any]], max_line_length: int = 42, max_duration: float = 5.0) -> str:
    """
    Generate SRT subtitle format from transcript segments.
    
    Args:
        segments: List of segment dictionaries with start, end, and text keys
        max_line_length: Maximum character length per line before splitting
        max_duration: Maximum duration in seconds for a single subtitle
        
    Returns:
        String containing formatted SRT content
    """
    srt_content = []
    subtitle_index = 1
    
    for segment in segments:
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        text = segment.get("text", "").strip()
        
        # Skip empty segments
        if not text:
            continue
            
        # Split long segments by duration
        if end_time - start_time > max_duration:
            # Calculate number of parts needed
            num_parts = int((end_time - start_time) / max_duration) + 1
            duration_per_part = (end_time - start_time) / num_parts
            
            # Split the segment text roughly by word count
            words = text.split()
            words_per_part = len(words) // num_parts
            
            for i in range(num_parts):
                part_start = start_time + (i * duration_per_part)
                part_end = part_start + duration_per_part
                
                # Get subset of words for this part
                start_idx = i * words_per_part
                end_idx = start_idx + words_per_part if i < num_parts - 1 else len(words)
                part_text = " ".join(words[start_idx:end_idx])
                
                # Format for SRT
                srt_content.append(f"{subtitle_index}")
                srt_content.append(f"{format_timestamp_srt(part_start)} --> {format_timestamp_srt(part_end)}")
                
                # Split long lines
                if len(part_text) > max_line_length:
                    half_length = len(part_text) // 2
                    # Try to split at space
                    split_pos = part_text.rfind(" ", 0, half_length + 10)
                    if split_pos == -1:
                        split_pos = half_length
                    
                    srt_content.append(f"{part_text[:split_pos]}")
                    srt_content.append(f"{part_text[split_pos:].strip()}")
                else:
                    srt_content.append(part_text)
                
                srt_content.append("")  # Empty line between entries
                subtitle_index += 1
        else:
            # Format for SRT
            srt_content.append(f"{subtitle_index}")
            srt_content.append(f"{format_timestamp_srt(start_time)} --> {format_timestamp_srt(end_time)}")
            
            # Split long lines
            if len(text) > max_line_length:
                half_length = len(text) // 2
                # Try to split at space
                split_pos = text.rfind(" ", 0, half_length + 10)
                if split_pos == -1:
                    split_pos = half_length
                
                srt_content.append(f"{text[:split_pos]}")
                srt_content.append(f"{text[split_pos:].strip()}")
            else:
                srt_content.append(text)
            
            srt_content.append("")  # Empty line between entries
            subtitle_index += 1
    
    return "\n".join(srt_content)


def generate_vtt(segments: list[dict[str, Any]], max_line_length: int = 42, max_duration: float = 5.0) -> str:
    """
    Generate WebVTT subtitle format from transcript segments.
    
    Args:
        segments: List of segment dictionaries with start, end, and text keys
        max_line_length: Maximum character length per line before splitting
        max_duration: Maximum duration in seconds for a single subtitle
        
    Returns:
        String containing formatted VTT content
    """
    vtt_content = ["WEBVTT", ""]  # VTT header
    
    for i, segment in enumerate(segments):
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        text = segment.get("text", "").strip()
        
        # Skip empty segments
        if not text:
            continue
            
        # Split long segments by duration
        if end_time - start_time > max_duration:
            # Calculate number of parts needed
            num_parts = int((end_time - start_time) / max_duration) + 1
            duration_per_part = (end_time - start_time) / num_parts
            
            # Split the segment text roughly by word count
            words = text.split()
            words_per_part = len(words) // num_parts
            
            for i in range(num_parts):
                part_start = start_time + (i * duration_per_part)
                part_end = part_start + duration_per_part
                
                # Get subset of words for this part
                start_idx = i * words_per_part
                end_idx = start_idx + words_per_part if i < num_parts - 1 else len(words)
                part_text = " ".join(words[start_idx:end_idx])
                
                # Format for VTT
                vtt_content.append(f"{format_timestamp_vtt(part_start)} --> {format_timestamp_vtt(part_end)}")
                
                # Split long lines
                if len(part_text) > max_line_length:
                    half_length = len(part_text) // 2
                    # Try to split at space
                    split_pos = part_text.rfind(" ", 0, half_length + 10)
                    if split_pos == -1:
                        split_pos = half_length
                    
                    vtt_content.append(f"{part_text[:split_pos]}")
                    vtt_content.append(f"{part_text[split_pos:].strip()}")
                else:
                    vtt_content.append(part_text)
                
                vtt_content.append("")  # Empty line between entries
        else:
            # Format for VTT
            vtt_content.append(f"{format_timestamp_vtt(start_time)} --> {format_timestamp_vtt(end_time)}")
            
            # Split long lines
            if len(text) > max_line_length:
                half_length = len(text) // 2
                # Try to split at space
                split_pos = text.rfind(" ", 0, half_length + 10)
                if split_pos == -1:
                    split_pos = half_length
                
                vtt_content.append(f"{text[:split_pos]}")
                vtt_content.append(f"{text[split_pos:].strip()}")
            else:
                vtt_content.append(text)
            
            vtt_content.append("")  # Empty line between entries
    
    return "\n".join(vtt_content)


def save_subtitles(transcript_result: dict[str, Any], output_path: str, format_type: str = "srt") -> str | None:
    """
    Save transcript with timestamps to subtitle file.
    
    Args:
        transcript_result: Dictionary with segments containing timestamp information
        output_path: Path where to save the subtitle file
        format_type: Format type, either 'srt' or 'vtt'
        
    Returns:
        Path to saved subtitle file or None if failed
    """
    logger = get_logger(__name__)
    segments = transcript_result.get("segments", [])
    
    if not segments:
        logger.error("No segments with timestamps found in transcript")
        return None
    
    # Generate appropriate extension if not specified
    base, ext = os.path.splitext(output_path)
    if not ext or ext.lower() not in ['.srt', '.vtt']:
        output_path = f"{base}.{format_type}"
    
    # Generate content based on format
    if format_type.lower() == "srt" or output_path.lower().endswith('.srt'):
        content = generate_srt(segments)
    elif format_type.lower() == "vtt" or output_path.lower().endswith('.vtt'):
        content = generate_vtt(segments)
    else:
        logger.error(f"Unsupported subtitle format: {format_type}")
        return None
    
    # Write to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Subtitle file saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving subtitle file: {str(e)}")
        return None

#fin