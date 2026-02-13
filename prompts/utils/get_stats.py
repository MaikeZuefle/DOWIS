#!/usr/bin/env python3
"""
Analyze audio prompts across multiple languages.
Generates statistics on duration by language, speaker, and task.
"""

import os
import wave
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import re


def get_wav_duration(filepath: str) -> float:
    """
    Get duration of a WAV file in seconds.
    
    Args:
        filepath: Path to the WAV file
        
    Returns:
        Duration in seconds
    """
    try:
        with wave.open(filepath, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        raise Exception(f"Error reading {filepath}: {str(e)}")


def parse_filename(filename: str) -> Tuple[str, str, str, str, str]:
    """
    Parse audio filename to extract components.
    Expected format: {language}_{speaker}_{task}_{style}_{number}.wav
    
    Args:
        filename: Name of the audio file
        
    Returns:
        Tuple of (language, speaker, task, style, number)
    """
    # Remove .wav extension
    name = filename.replace('.wav', '')
    
    # Expected pattern: language_speaker_task_style_number
    pattern = r'^([a-z]{2})_(female[12]|male[12])_([a-z0-9]+)_([a-z]+)_(\d+)$'
    match = re.match(pattern, name)
    
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    
    language, speaker, task, style, number = match.groups()
    return language, speaker, task, style, number


def seconds_to_hms(seconds: float) -> Dict[str, float]:
    """
    Convert seconds to hours, minutes, and seconds.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Dictionary with hours, minutes, seconds
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    return {
        "hours": hours,
        "minutes": minutes,
        "seconds": round(secs, 2),
        "total_seconds": round(seconds, 2)
    }


def analyze_audio_prompts(base_dir: str) -> Dict:
    """
    Analyze all audio prompts in the directory structure.
    
    Args:
        base_dir: Base directory containing language subdirectories
        
    Returns:
        Dictionary with comprehensive statistics
    """
    base_path = Path(base_dir)
    
    # Data structures for statistics
    language_durations = defaultdict(float)
    speaker_durations = defaultdict(lambda: defaultdict(float))  # language -> speaker -> duration
    task_durations = defaultdict(list)  # task -> list of durations
    
    # Track processed files and errors
    processed_files = []
    edge_cases = []
    total_files = 0
    
    # Iterate through language directories
    for lang_dir in sorted(base_path.iterdir()):
        if not lang_dir.is_dir():
            continue
            
        language_code = lang_dir.name
        
        # Process all WAV files in this language directory
        wav_files = list(lang_dir.glob('*.wav'))
        total_files += len(wav_files)
        
        for wav_file in wav_files:
            try:
                # Parse filename
                language, speaker, task, style, number = parse_filename(wav_file.name)
                
                # Verify language code matches directory
                if language != language_code:
                    edge_cases.append({
                        "file": str(wav_file),
                        "issue": f"Language code mismatch: filename has '{language}' but in '{language_code}' directory"
                    })
                
                # Get duration
                duration = get_wav_duration(str(wav_file))
                
                # Update statistics
                language_durations[language] += duration
                speaker_durations[language][speaker] += duration
                task_durations[task].append(duration)
                
                processed_files.append({
                    "file": wav_file.name,
                    "language": language,
                    "speaker": speaker,
                    "task": task,
                    "style": style,
                    "number": number,
                    "duration_seconds": round(duration, 2)
                })
                
            except ValueError as e:
                edge_cases.append({
                    "file": str(wav_file),
                    "issue": f"Filename parsing error: {str(e)}"
                })
            except Exception as e:
                edge_cases.append({
                    "file": str(wav_file),
                    "issue": f"Processing error: {str(e)}"
                })
    
    # Calculate total duration across all files
    total_duration_seconds = sum(language_durations.values())
    
    # Calculate total average per speaker (across all language-speaker combinations)
    # Count unique (language, speaker) pairs
    speaker_count = 0
    total_speaker_duration = 0
    for language in speaker_durations:
        for speaker in speaker_durations[language]:
            speaker_count += 1
            total_speaker_duration += speaker_durations[language][speaker]
    
    avg_per_speaker_seconds = total_speaker_duration / speaker_count if speaker_count > 0 else 0
    
    # Calculate statistics
    stats = {
        "summary": {
            "total_files_found": total_files,
            "successfully_processed": len(processed_files),
            "edge_cases": len(edge_cases),
            "total_duration_all_files": seconds_to_hms(total_duration_seconds),
            "total_language_speaker_combinations": speaker_count,
            "average_duration_per_speaker_per_language": seconds_to_hms(avg_per_speaker_seconds)
        },
        "by_language": {},
        "by_speaker_per_language": {},
        "by_task_across_all_languages": {},
        "edge_cases": edge_cases
    }
    
    # Language statistics
    for language, total_seconds in sorted(language_durations.items()):
        stats["by_language"][language] = seconds_to_hms(total_seconds)
    
    # Speaker statistics per language
    for language in sorted(speaker_durations.keys()):
        stats["by_speaker_per_language"][language] = {}
        speakers = speaker_durations[language]
        
        for speaker, total_seconds in sorted(speakers.items()):
            stats["by_speaker_per_language"][language][speaker] = seconds_to_hms(total_seconds)
        
        # Calculate average per speaker for this language
        if speakers:
            avg_seconds = sum(speakers.values()) / len(speakers)
            stats["by_speaker_per_language"][language]["average_per_speaker"] = seconds_to_hms(avg_seconds)
    
    # Task statistics across all languages
    for task, durations in sorted(task_durations.items()):
        if durations:
            avg_seconds = sum(durations) / len(durations)
            total_seconds = sum(durations)
            stats["by_task_across_all_languages"][task] = {
                "average_duration": seconds_to_hms(avg_seconds),
                "total_duration": seconds_to_hms(total_seconds),
                "count": len(durations)
            }
    
    return stats


def main():
    """Main execution function."""
    # Assuming script is run from audio_prompts directory or parent
    # Try to find audio_prompts directory
    current_dir = Path.cwd()
    
    if (current_dir / "audio_prompts").exists():
        base_dir = current_dir / "audio_prompts"
    elif current_dir.name == "audio_prompts":
        base_dir = current_dir
    else:
        print("Error: Could not find 'audio_prompts' directory")
        print(f"Current directory: {current_dir}")
        print("Please run this script from the parent directory of 'audio_prompts' or from within 'audio_prompts'")
        return
    
    print(f"Analyzing audio files in: {base_dir}")
    print("This may take a while...\n")
    
    # Analyze the audio prompts
    stats = analyze_audio_prompts(str(base_dir))
    
    # Save to JSON file
    output_file = "audio_prompts_statistics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Analysis complete!")
    print(f"✓ Statistics saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Total files found: {stats['summary']['total_files_found']}")
    print(f"  Successfully processed: {stats['summary']['successfully_processed']}")
    print(f"  Edge cases found: {stats['summary']['edge_cases']}")
    
    # Display total duration
    total_dur = stats['summary']['total_duration_all_files']
    print(f"\n  Total duration of all files: {total_dur['hours']}h {total_dur['minutes']}m {total_dur['seconds']}s")
    
    # Display average per speaker
    avg_speaker = stats['summary']['average_duration_per_speaker_per_language']
    print(f"  Total language-speaker combinations: {stats['summary']['total_language_speaker_combinations']}")
    print(f"  Average duration per speaker per language: {avg_speaker['hours']}h {avg_speaker['minutes']}m {avg_speaker['seconds']}s")
    
    if stats['summary']['edge_cases'] > 0:
        print(f"\n⚠ Warning: {stats['summary']['edge_cases']} edge cases detected. See JSON output for details.")


if __name__ == "__main__":
    main()