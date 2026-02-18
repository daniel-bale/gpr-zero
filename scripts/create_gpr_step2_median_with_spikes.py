#!/usr/bin/env python3
"""
Create GPR Step2 median plot with spike markers for thesis.
Shows all Step2 spikes (not Lau's) with vertical shaded regions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from src.gpr_zero.utils.font_setup import ensure_cjk_font_setup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup CJK font support
ensure_cjk_font_setup()

def load_spike_data():
    """Load Step2 spike data from unified detection."""
    spike_file = Path("data/gpr_spikes/unified_detection_3outlets/step2_median_spikes.csv")
    if not spike_file.exists():
        logger.warning(f"Spike file not found: {spike_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(spike_file)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['peak_date'] = pd.to_datetime(df['peak_date'])
    
    # Filter out 2025 spikes (only show spikes up to 2024)
    df = df[df['peak_date'] <= '2024-12-31']
    
    logger.info(f"Loaded {len(df)} Step2 spikes (3-outlet corrected, excluding 2025)")
    return df

def fold_short_answer(text):
    """Folds a short text for better plot display - at least 2 lines, max 3 words per line."""
    if pd.isna(text) or not text:
        return ""
    
    # First split on commas if they exist
    if ', ' in text:
        parts = text.split(', ')
        # Don't split each part further - just join with newlines
        return ',\n'.join(parts)
    
    # Otherwise split single phrase
    return split_into_lines(text)

def split_into_lines(text):
    """Split text into lines with at least 2 lines, max 3 words per line."""
    words = text.split()
    
    if len(words) <= 2:
        # If 2 or fewer words, split into 2 lines (1 word each)
        if len(words) == 1:
            return text  # Can't split 1 word
        return f"{words[0]}\n{' '.join(words[1:])}"
    
    elif len(words) <= 6:
        # For 3-6 words, split into 2 lines
        mid = len(words) // 2
        return f"{' '.join(words[:mid])}\n{' '.join(words[mid:])}"
    
    else:
        # For more than 6 words, split into 3 lines (3 words per line max)
        third = len(words) // 3
        return f"{' '.join(words[:third])}\n{' '.join(words[third:2*third])}\n{' '.join(words[2*third:])}"

def create_spike_labels(all_spikes):
    """Create event labels for all spikes from CSV summary."""
    # Load spike summaries from CSV
    summary_file = Path("data/gpr_spikes/step2_all_spikes_summary.csv")
    if not summary_file.exists():
        logger.warning(f"Spike summary file not found: {summary_file}")
        # Fallback to date-based labels
        all_spikes['spike_label'] = all_spikes['peak_date'].dt.strftime('%Y-%m')
        return all_spikes
    
    summary_df = pd.read_csv(summary_file)
    summary_df['peak_date'] = pd.to_datetime(summary_df['peak_date'])
    
    # Merge event summaries
    all_spikes = all_spikes.merge(
        summary_df[['peak_date', 'event_summary']], 
        on='peak_date', 
        how='left'
    )
    
    # Handle missing summaries with specific labels for known spikes
    missing_labels = {
        '2020-11-02': 'US Election 2020',
        '2023-01-30': 'Chinese Balloon Incident'
    }
    
    # Apply fold_short_answer to event summaries and handle missing values
    all_spikes['spike_label'] = all_spikes.apply(
        lambda row: fold_short_answer(row['event_summary']) if pd.notna(row['event_summary']) 
        else missing_labels.get(row['peak_date'].strftime('%Y-%m-%d'), row['peak_date'].strftime('%Y-%m')),
        axis=1
    )
    
    return all_spikes

def main():
    """Create GPR Step2 median plot with spike markers."""
    
    # Load the corrected 3-outlet median data
    median_file = Path("data/gpr_indices_3outlets/step2_median_gpr_3outlets.csv")
    df = pd.read_csv(median_file)
    df['date'] = pd.to_datetime(df['week'])
    
    # Use the scaled values and calculate percentiles
    df['median_scaled'] = df['gpr_index_scaled']
    
    # For confidence bands, we need to load all Step2 time series
    timeseries_dir = Path("/home5/chuenho/gpr-zero/data/gpr_timeseries_3outlets")
    step2_files = list(timeseries_dir.glob("step2_*_W_overall_3outlets.parquet"))
    
    logger.info(f"Loading {len(step2_files)} Step2 files for confidence bands...")
    
    # Load all series
    all_series = []
    for file in sorted(step2_files[:40]):  # First 40 Step2 files
        df_temp = pd.read_parquet(file)
        df_temp['date'] = pd.to_datetime(df_temp['week'])
        all_series.append(df_temp[['date', 'gpr_index']])
    
    # Combine and calculate percentiles
    combined_df = all_series[0].copy()
    combined_df.rename(columns={'gpr_index': 'gpr_index_0'}, inplace=True)
    
    for i, df_temp in enumerate(all_series[1:], 1):
        combined_df = combined_df.merge(df_temp, on='date')
        combined_df.rename(columns={'gpr_index': f'gpr_index_{i}'}, inplace=True)
    
    gpr_cols = [col for col in combined_df.columns if col.startswith('gpr_index')]
    
    # Calculate percentiles and scale them
    mean_value = combined_df[gpr_cols].median(axis=1).mean()
    df = df.merge(combined_df[['date']], on='date', how='left')
    
    # Calculate scaled percentiles
    for date in df['date']:
        if date in combined_df['date'].values:
            row_data = combined_df[combined_df['date'] == date][gpr_cols].values[0]
            df.loc[df['date'] == date, 'p025_scaled'] = np.percentile(row_data, 2.5) / mean_value * 100
            df.loc[df['date'] == date, 'p975_scaled'] = np.percentile(row_data, 97.5) / mean_value * 100
            df.loc[df['date'] == date, 'p16_scaled'] = np.percentile(row_data, 16) / mean_value * 100
            df.loc[df['date'] == date, 'p84_scaled'] = np.percentile(row_data, 84) / mean_value * 100
    
    logger.info(f"Loaded corrected 3-outlet median data with {len(df)} weeks")
    logger.info(f"Mean: {df['median_scaled'].mean():.2f}, Std: {df['median_scaled'].std():.2f}")
    
    # Filter to 2017-2024 range
    df = df[(df['date'] >= '2017-01-01') & (df['date'] <= '2024-12-31')]
    
    # Load spike data
    all_spikes = load_spike_data()
    all_spikes = create_spike_labels(all_spikes)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Remove spike regions for cleaner look
    # (commented out but kept for reference)
    # spike_color = '#ffcccc'  # Light red for all spikes
    # for _, spike in all_spikes.iterrows():
    #     # Extend end date by 6 days for weekly spikes
    #     actual_end = spike['end_date'] + pd.Timedelta(days=6)
    #     ax.axvspan(spike['start_date'], actual_end, 
    #                alpha=0.3, color=spike_color, zorder=1)
    
    # Plot only the median GPR line (no confidence bands for cleaner look)
    ax.plot(df['date'], df['median_scaled'], color='darkblue', 
            linewidth=2.5, label='Median GPR', zorder=4)
    
    # Add markers and labels for ALL spike peaks
    if not all_spikes.empty:
        spike_dates = []
        spike_values = []
        
        # Get peak values from the spike data
        for _, spike in all_spikes.iterrows():
            # Find the peak value from the median data
            peak_week = df[df['date'] == spike['peak_date']]
            if not peak_week.empty:
                peak_value_scaled = peak_week['median_scaled'].values[0]
            else:
                # If exact week not found, find closest
                closest_idx = (df['date'] - spike['peak_date']).abs().idxmin()
                peak_value_scaled = df.loc[closest_idx, 'median_scaled']
            
            # Store for marker plotting
            spike_dates.append(spike['peak_date'])
            spike_values.append(peak_value_scaled)
        
        # First add the spike markers in blue color similar to line (smaller size)
        ax.scatter(spike_dates, spike_values, 
                  color='steelblue', s=20, zorder=5, marker='o', 
                  edgecolors='darkblue', linewidth=0.5)
        
        # Manual label positioning to avoid overlaps
        # Group spikes by proximity and assign vertical levels
        spike_info = []
        for i, (_, spike) in enumerate(all_spikes.iterrows()):
            spike_info.append({
                'date': spike['peak_date'],
                'value': spike_values[i],
                'label': spike['spike_label'],
                'index': i
            })
        
        # Sort by date
        spike_info.sort(key=lambda x: x['date'])
        
        # Debug: print all labels
        logger.info("All spike labels:")
        for info in spike_info:
            logger.info(f"  Date: {info['date']}, Label: '{info['label']}'")
        
        # Define manual y-offsets for different vertical tiers
        # Higher tier = further from line - increased spacing
        tier_offsets = [40, 100, 160, 220, 280, 340, 400]  # Offset from spike value
        
        # Assign tiers based on proximity (if spikes are within 180 days, put on different tiers)
        spike_tiers = [0] * len(spike_info)
        
        for i in range(len(spike_info)):
            # Find the lowest available tier that doesn't conflict with nearby spikes
            for tier in range(len(tier_offsets)):
                conflict = False
                for j in range(len(spike_info)):
                    if i != j and spike_tiers[j] == tier:
                        # Check if dates are too close
                        days_apart = abs((spike_info[i]['date'] - spike_info[j]['date']).days)
                        if days_apart < 180:  # 6 months
                            conflict = True
                            break
                if not conflict:
                    spike_tiers[i] = tier
                    break
        
        # No bbox - text without frames
        bbox_dict = None
        
        # First pass: draw all texts and get their bounding boxes
        text_objects = []
        bboxes = []
        
        for i, info in enumerate(spike_info):
            tier = spike_tiers[i]
            y_offset = tier_offsets[tier % len(tier_offsets)]
            
            # Position text above the spike with tier-based offset
            text_y = info['value'] + y_offset
            
            # Manual adjustments for specific labels to avoid line overlaps
            ha = 'center'
            x_offset = pd.Timedelta(days=0)
            
            # Check label without newlines for matching
            label_clean = info['label'].replace('\n', ' ')
            
            # Debug logging
            if 'Cross-Strait' in label_clean or 'IPEF' in label_clean:
                logger.debug(f"Label: {label_clean}, Date: {info['date']}, Year: {info['date'].year}")
            
            if 'Cross-Strait unification pressure' in label_clean:
                ha = 'left'
                x_offset = pd.Timedelta(days=240)  # Move right even more
                logger.info(f"Moving Cross-Strait unification pressure right by 240 days (date: {info['date']})")
            elif 'Cross-Strait military escalation' in label_clean:  # Handle all instances
                ha = 'left' 
                x_offset = pd.Timedelta(days=240)  # Move right even more
                logger.info(f"Moving Cross-Strait military escalation right by 240 days (date: {info['date']}, year: {info['date'].year})")
            elif 'IPEF exclusion' in label_clean:
                ha = 'right'
                x_offset = pd.Timedelta(days=0)  # No offset
                logger.info(f"IPEF exclusion with right alignment and 0 offset (date: {info['date']})")
            
            # Draw text with adjustments
            text_obj = ax.text(info['date'] + x_offset, text_y, info['label'],
                              ha=ha, va='bottom', fontsize=8,
                              bbox=bbox_dict, zorder=6)
            text_objects.append(text_obj)
        
        # Force draw to get accurate bounding boxes
        fig.canvas.draw()
        
        # Get bounding boxes in data coordinates
        for text_obj in text_objects:
            bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_data = bbox.transformed(ax.transData.inverted())
            bboxes.append(bbox_data)
        
        # Brute-force overlap elimination
        max_iterations = 50  # More iterations
        for iteration in range(max_iterations):
            fig.canvas.draw()
            overlap_found = False
            
            # Update all bounding boxes
            for idx, text_obj in enumerate(text_objects):
                bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
                bboxes[idx] = bbox.transformed(ax.transData.inverted())
            
            # Check all pairs
            for i in range(len(spike_info)):
                for j in range(i + 1, len(spike_info)):
                    bbox1 = bboxes[i]
                    bbox2 = bboxes[j]
                    
                    # Calculate overlap with padding
                    padding = 10  # Generous padding
                    x_overlap = (bbox1.x0 - padding < bbox2.x1 + padding and bbox1.x1 + padding > bbox2.x0 - padding)
                    y_overlap = (bbox1.y0 - padding < bbox2.y1 + padding and bbox1.y1 + padding > bbox2.y0 - padding)
                    
                    if x_overlap and y_overlap:
                        overlap_found = True
                        # Always move the one with higher current y position
                        y1 = text_objects[i].get_position()[1]
                        y2 = text_objects[j].get_position()[1]
                        
                        if y1 > y2:
                            # Move i up by a fixed amount
                            text_objects[i].set_position((spike_info[i]['date'], y1 + 40))
                        else:
                            # Move j up by a fixed amount
                            text_objects[j].set_position((spike_info[j]['date'], y2 + 40))
                        
                        # Only fix one overlap per iteration to avoid cascading issues
                        break
                if overlap_found:
                    break
            
            if not overlap_found:
                logger.info(f"  No overlaps found after {iteration + 1} iterations")
                break
        else:
            logger.warning(f"  Reached max iterations ({max_iterations}) in overlap adjustment")
        
        # Store connector line data AFTER all adjustments
        connector_data = []
        for i, (info, text_obj) in enumerate(zip(spike_info, text_objects)):
            connector_data.append({
                'spike_date': info['date'],
                'spike_value': info['value'],
                'text_obj': text_obj
            })
    
    # Horizontal line at 100
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Remove title and labels (for clean thesis figure)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    
    # Set x-axis limits to 2017-2024
    ax.set_xlim(pd.Timestamp('2017-01-01'), pd.Timestamp('2024-12-31'))
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', fontsize=10, frameon=False)
    
    # Tight layout
    plt.tight_layout()
    
    # Draw spike labels and handle overlaps
    if not all_spikes.empty:
        # Final overlap check and adjustment with debugging
        logger.info("Checking for text overlaps after initial positioning...")
        for iteration in range(20):  # More iterations for complex adjustments
            overlap_found = False
            adjustments_made = []
            
            # Check text-to-text overlaps
            for i in range(len(text_objects)):
                for j in range(i + 1, len(text_objects)):
                    bbox1 = text_objects[i].get_window_extent(renderer=fig.canvas.get_renderer())
                    bbox2 = text_objects[j].get_window_extent(renderer=fig.canvas.get_renderer())
                    bbox1_data = bbox1.transformed(ax.transData.inverted())
                    bbox2_data = bbox2.transformed(ax.transData.inverted())
                    
                    # Use date number for x-coordinates to ensure proper comparison
                    date1 = mdates.date2num(text_objects[i].get_position()[0])
                    date2 = mdates.date2num(text_objects[j].get_position()[0])
                    x1_min = date1 - 30  # Approximate text width in days
                    x1_max = date1 + 30
                    x2_min = date2 - 30
                    x2_max = date2 + 30
                    
                    padding = 15  # Increased padding
                    x_overlap = (x1_min - padding < x2_max + padding and 
                               x1_max + padding > x2_min - padding)
                    y_overlap = (bbox1_data.y0 - padding < bbox2_data.y1 + padding and 
                               bbox1_data.y1 + padding > bbox2_data.y0 - padding)
                    
                    if x_overlap and y_overlap:
                        overlap_found = True
                        # Move the one with higher y position up more
                        y1 = text_objects[i].get_position()[1]
                        y2 = text_objects[j].get_position()[1]
                        if y1 > y2:
                            new_y = bbox2_data.y1 + padding + (bbox1_data.y1 - bbox1_data.y0)
                            text_objects[i].set_position((text_objects[i].get_position()[0], new_y))
                            adjustments_made.append(f"Moved text {i} up to {new_y:.1f}")
                        else:
                            new_y = bbox1_data.y1 + padding + (bbox2_data.y1 - bbox2_data.y0)
                            text_objects[j].set_position((text_objects[j].get_position()[0], new_y))
                            adjustments_made.append(f"Moved text {j} up to {new_y:.1f}")
                        fig.canvas.draw()
            
            # Check text-to-line overlaps (with median GPR line)
            for i, text_obj in enumerate(text_objects):
                bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
                bbox_data = bbox.transformed(ax.transData.inverted())
                
                # Get the text position date
                text_date = text_obj.get_position()[0]
                
                # Find the median GPR value at this date
                if text_date in df['date'].values:
                    median_value = df[df['date'] == text_date]['median_scaled'].values[0]
                else:
                    # Interpolate if exact date not found
                    closest_idx = (df['date'] - text_date).abs().idxmin()
                    median_value = df.loc[closest_idx, 'median_scaled']
                
                # Also check nearby dates for the line
                date_range = pd.date_range(text_date - pd.Timedelta(days=14), 
                                         text_date + pd.Timedelta(days=14), freq='W-MON')
                max_median_nearby = 0
                for check_date in date_range:
                    if check_date in df['date'].values:
                        nearby_value = df[df['date'] == check_date]['median_scaled'].values[0]
                        max_median_nearby = max(max_median_nearby, nearby_value)
                
                # Check if text box overlaps with the median line
                line_padding = 20  # Extra padding for line visibility
                if bbox_data.y0 < max_median_nearby + line_padding:
                    overlap_found = True
                    # Move text up to clear the line
                    new_y = max_median_nearby + line_padding + (bbox_data.y1 - bbox_data.y0)
                    text_obj.set_position((text_date, new_y))
                    fig.canvas.draw()
            
            if adjustments_made:
                logger.info(f"  Iteration {iteration}: {len(adjustments_made)} adjustments made")
            
            if not overlap_found:
                logger.info(f"  No overlaps found after {iteration + 1} iterations")
                break
        
        # Update y-axis limits to include all text
        y_max = 0
        for text_obj in text_objects:
            bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_data = bbox.transformed(ax.transData.inverted())
            y_max = max(y_max, bbox_data.y1)
        
        ax.set_ylim(0, max(y_max + 50, df['p975_scaled'].max() * 1.2))
        
        # FINAL OVERLAP CHECK after everything is complete
        logger.info("\nPerforming final overlap check after all modifications...")
        fig.canvas.draw()
        
        overlap_report = []
        # Check all pairs of text objects
        for i in range(len(text_objects)):
            for j in range(i + 1, len(text_objects)):
                bbox1 = text_objects[i].get_window_extent(renderer=fig.canvas.get_renderer())
                bbox2 = text_objects[j].get_window_extent(renderer=fig.canvas.get_renderer())
                
                # Check for overlap with some tolerance
                tolerance = 2  # pixels
                if (bbox1.x0 - tolerance < bbox2.x1 + tolerance and 
                    bbox1.x1 + tolerance > bbox2.x0 - tolerance and
                    bbox1.y0 - tolerance < bbox2.y1 + tolerance and 
                    bbox1.y1 + tolerance > bbox2.y0 - tolerance):
                    
                    label1 = spike_info[i]['label'].replace('\n', ' ')
                    label2 = spike_info[j]['label'].replace('\n', ' ')
                    overlap_report.append(f"  OVERLAP FOUND: '{label1}' overlaps with '{label2}'")
                    overlap_report.append(f"    Box 1: x=({bbox1.x0:.1f}, {bbox1.x1:.1f}), y=({bbox1.y0:.1f}, {bbox1.y1:.1f})")
                    overlap_report.append(f"    Box 2: x=({bbox2.x0:.1f}, {bbox2.x1:.1f}), y=({bbox2.y0:.1f}, {bbox2.y1:.1f})")
        
        if overlap_report:
            logger.warning("Final overlap check found issues:")
            for line in overlap_report:
                logger.warning(line)
            
            # Brute-force fix for any remaining overlaps
            logger.info("Brute-force fixing remaining overlaps...")
            for attempt in range(100):  # Many more iterations
                fig.canvas.draw()
                fixed_any = False
                
                for i in range(len(text_objects)):
                    for j in range(i + 1, len(text_objects)):
                        bbox1 = text_objects[i].get_window_extent(renderer=fig.canvas.get_renderer())
                        bbox2 = text_objects[j].get_window_extent(renderer=fig.canvas.get_renderer())
                        
                        # Check for overlap with very small tolerance
                        tolerance = 5  # Increased tolerance
                        if (bbox1.x0 - tolerance < bbox2.x1 + tolerance and 
                            bbox1.x1 + tolerance > bbox2.x0 - tolerance and
                            bbox1.y0 - tolerance < bbox2.y1 + tolerance and 
                            bbox1.y1 + tolerance > bbox2.y0 - tolerance):
                            
                            # Move the higher one up
                            y1 = text_objects[i].get_position()[1]
                            y2 = text_objects[j].get_position()[1]
                            
                            if y1 > y2:
                                # Move i up
                                new_y = y1 + 50
                                text_objects[i].set_position((text_objects[i].get_position()[0], new_y))
                            else:
                                # Move j up
                                new_y = y2 + 50
                                text_objects[j].set_position((text_objects[j].get_position()[0], new_y))
                            
                            fixed_any = True
                            break
                    if fixed_any:
                        break
                
                if not fixed_any:
                    logger.info(f"  All overlaps resolved after {attempt + 1} attempts")
                    break
            
            # Final check after fixes
            fig.canvas.draw()
            final_overlaps = []
            for i in range(len(text_objects)):
                for j in range(i + 1, len(text_objects)):
                    bbox1 = text_objects[i].get_window_extent(renderer=fig.canvas.get_renderer())
                    bbox2 = text_objects[j].get_window_extent(renderer=fig.canvas.get_renderer())
                    
                    tolerance = 2
                    if (bbox1.x0 - tolerance < bbox2.x1 + tolerance and 
                        bbox1.x1 + tolerance > bbox2.x0 - tolerance and
                        bbox1.y0 - tolerance < bbox2.y1 + tolerance and 
                        bbox1.y1 + tolerance > bbox2.y0 - tolerance):
                        
                        label1 = spike_info[i]['label'].replace('\n', ' ')
                        label2 = spike_info[j]['label'].replace('\n', ' ')
                        final_overlaps.append(f"{label1} <-> {label2}")
            
            if final_overlaps:
                logger.warning(f"Still {len(final_overlaps)} overlaps remaining after fixes")
            else:
                logger.info("All overlaps successfully resolved!")
            
            # Update y-axis limits again after final adjustments
            y_max = 0
            for text_obj in text_objects:
                bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
                bbox_data = bbox.transformed(ax.transData.inverted())
                y_max = max(y_max, bbox_data.y1)
            
            ax.set_ylim(0, max(y_max + 50, df['p975_scaled'].max() * 1.2))
        else:
            logger.info("  Final check: No overlaps detected!")
        
        # Draw connector lines AFTER all text positioning is complete
        logger.info("Drawing connector lines after final positioning...")
        fig.canvas.draw()
        
        for i, data in enumerate(connector_data):
            text_obj = data['text_obj']
            
            # Get FINAL text position after all adjustments
            text_pos = text_obj.get_position()
            text_x = text_pos[0]
            text_y = text_pos[1]
            
            # Since va='bottom', the text_y is already the bottom of the text
            text_bottom_y = text_y
            
            # Get text alignment to determine connection point
            ha = text_obj.get_ha()
            
            # Draw connecting line from spike to bottom of text
            ax.plot([data['spike_date'], text_x], 
                   [data['spike_value'], text_bottom_y],
                   color='darkgray', linewidth=1.0, alpha=0.8, zorder=3,
                   linestyle='-', marker='', solid_capstyle='round')
    
    # Save the plot
    output_file = Path("data/gpr_indices_3outlets/gpr_step2_median_with_spikes_clean.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    plt.close()
    
    # Copy to thesis figures
    import shutil
    thesis_file = Path("/home5/chuenho/thesis/figures/gpr_step2_median_with_spikes_clean.png")
    shutil.copy(output_file, thesis_file)
    logger.info(f"Copied to thesis figures: {thesis_file}")
    
    # Log spike statistics
    logger.info(f"\nSpike Statistics:")
    logger.info(f"Total Step2 spikes shown: {len(all_spikes)}")
    
    if not all_spikes.empty:
        logger.info(f"\nStep2 spike dates:")
        for _, spike in all_spikes.iterrows():
            logger.info(f"  {spike['start_date'].strftime('%Y-%m-%d')} to {spike['end_date'].strftime('%Y-%m-%d')} (peak: {spike['peak_date'].strftime('%Y-%m-%d')})")

if __name__ == "__main__":
    main()