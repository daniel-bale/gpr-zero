import duckdb
import polars as pl
import pathlib
import argparse # Added for command-line arguments

# --- Argument Parsing ---
# This section is added to define and parse command-line arguments.
parser = argparse.ArgumentParser(description="Compare two DuckDB databases containing news articles.")
parser.add_argument("new_db_path", type=str, help="Path to the NEW database file.")
parser.add_argument("old_db_path", type=str, help="Path to the OLD database file to compare against.")
args = parser.parse_args()

# Use paths from arguments
new_db_path_str = args.new_db_path
old_db_path_str = args.old_db_path
# --- End of Argument Parsing modification ---

new_db_path = pathlib.Path(new_db_path_str)
old_db_path = pathlib.Path(old_db_path_str)

# --- Helper function to print snippets ---
def print_comparison(title, old_content, new_content):
    print(f"\n--- Comparing Title: {title} ---")
    
    print("[OLD DB Content Snippet]")
    if old_content:
        old_snippet = old_content.strip().replace('\\n', ' ')[:300] + ("..." if len(old_content) > 300 else "")
        print(old_snippet)
    else:
        print("None")
        
    print("\n[NEW DB Content Snippet]")
    if new_content:
        new_snippet = new_content.strip().replace('\\n', ' ')[:300] + ("..." if len(new_content) > 300 else "")
        print(new_snippet)
    else:
        print("None")
    print("-" * 20)

# --- Main Comparison Logic ---
con_new = None
con_old = None

try:
    # Check existence
    print(f"Comparing NEW DB: {new_db_path_str} WITH OLD DB: {old_db_path_str}")
    
    print(f"Checking NEW DB: {new_db_path_str}")
    if not new_db_path.exists():
        print(f"Error: NEW Database file does not exist at {new_db_path_str}")
        exit()

    print(f"Checking OLD DB: {old_db_path_str}")
    if not old_db_path.exists():
        print(f"Error: OLD Database file does not exist at {old_db_path_str}")
        exit() 

    # Connect
    print("\nConnecting to databases...")
    con_new = duckdb.connect(database=new_db_path_str, read_only=True)
    con_old = duckdb.connect(database=old_db_path_str, read_only=True)
    print("Connections established.")

    # === Stage 1: Date Count Comparison ===
    print("\n======= Stage 1: Comparing Article Counts per Date =======")
    
    # Query Old DB Dates (Assuming 'datetime' column can be cast to DATE)
    try:
        print("Querying date counts from OLD DB...")
        q_old_dates = "SELECT datetime::DATE AS article_date, COUNT(*) AS count_old FROM articles GROUP BY article_date ORDER BY article_date"
        old_dates_df = con_old.execute(q_old_dates).pl()
        print(f"Found {len(old_dates_df)} distinct dates in OLD DB.")
    except Exception as e:
        print(f"Error querying OLD DB dates: {e}")
        old_dates_df = pl.DataFrame({"article_date": [], "count_old": []}) # Empty frame on error

    # Query New DB Dates (Casting 'warc_date' TIMESTAMP to DATE)
    try:
        print("Querying date counts from NEW DB...")
        q_new_dates = "SELECT warc_date::DATE AS article_date, COUNT(*) AS count_new FROM articles GROUP BY article_date ORDER BY article_date"
        new_dates_df = con_new.execute(q_new_dates).pl()
        print(f"Found {len(new_dates_df)} distinct dates in NEW DB.")
    except Exception as e:
        print(f"Error querying NEW DB dates: {e}")
        new_dates_df = pl.DataFrame({"article_date": [], "count_new": []}) # Empty frame on error
        
    # Compare date counts
    if not old_dates_df.is_empty() or not new_dates_df.is_empty():
        print("\nComparing counts (showing dates with differences or present in only one DB):")
        # Perform outer join
        comparison_df = old_dates_df.join(
            new_dates_df, on="article_date", how="outer"
        ).fill_null(0) # Fill missing counts with 0
        
        # Filter for differences
        diff_df = comparison_df.filter(pl.col("count_old") != pl.col("count_new"))
        
        if diff_df.is_empty():
            print("Counts per date appear consistent between databases.")
        else:
            print(diff_df)
    else:
         print("Could not perform date count comparison due to query errors or empty results.")

    # === Stage 2: Content Consistency Check ===
    print("\n======= Stage 2: Comparing Content for Matching Titles =======")
    
    try:
        print("Querying articles with non-null titles/content from OLD DB...")
        # Ensure title and content are not null for meaningful comparison
        q_old_content = "SELECT DISTINCT title, content AS content_old FROM articles WHERE title IS NOT NULL AND content IS NOT NULL"
        old_content_df = con_old.execute(q_old_content).pl()
        print(f"Found {len(old_content_df)} potential articles in OLD DB.")

        print("Querying articles with non-null titles/content from NEW DB...")
        q_new_content = "SELECT DISTINCT title, content AS content_new FROM articles WHERE title IS NOT NULL AND content IS NOT NULL"
        new_content_df = con_new.execute(q_new_content).pl()
        print(f"Found {len(new_content_df)} potential articles in NEW DB.")

        if not old_content_df.is_empty() and not new_content_df.is_empty():
            print("\nJoining articles on matching titles (OLD:title vs NEW:title)...")
            # Join on titles
            joined_content_df = old_content_df.join(
                new_content_df, left_on="title", right_on="title", how="inner"
            )
            
            match_count = len(joined_content_df)
            print(f"Found {match_count} articles with matching titles.")

            if match_count > 0:
                sample_size = min(5, match_count) # Show up to 5 samples
                print(f"\nShowing content comparison for {sample_size} sample(s):")
                
                # Take a sample (using head after join for simplicity)
                sample_to_compare = joined_content_df.head(sample_size)
                
                for row in sample_to_compare.iter_rows(named=True):
                    print_comparison(row['title'], row['content_old'], row['content_new'])
            else:
                print("No articles with matching titles found for content comparison.")
        else:
            print("Skipping content comparison due to empty results from one or both databases.")

    except Exception as e:
        print(f"\nAn error occurred during content comparison: {e}")

except Exception as e:
    print(f"\nAn error occurred during database processing: {e}")
finally:
    if con_new:
        con_new.close()
        print("\nNEW Database connection closed.")
    if con_old:
        con_old.close()
        print("OLD Database connection closed.")

print("\nComparison script finished.") 