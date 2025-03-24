"""Utility functions for working with dictionaries in the GPR-Zero project."""


def transform_dict_to_search_strings(
    dictionary: dict, plain_text: bool = False
) -> dict:
    """
    Transform all dictionary keywords into search strings following specific rules:

    1. Prioritize processing of bracketed keywords:
       Format as "keyword1 and (keyword2 or keyword3)"

    2. Special handling for keywords with spaces or special characters:
       Enclose in quotation marks like "wisers.com" or "wisers.net"

    Args:
        dictionary: Dictionary containing keywords in the format:
                   {"metadata": {...}, "keywords_lists": {...}, "categories": {...}}
        plain_text: If True, returns strings with actual double quotes (not escaped)
                   for easier copy/paste. Note: This format is not valid JSON.

    Returns:
        Dictionary of search strings keyed by query type
    """
    search_strings = {}

    # Helper function to add quotes around keywords
    def format_keyword(keyword: str) -> str:
        # Check that keyword doesn't already contain double quotes
        assert '"' not in keyword, (
            f"Keyword '{keyword}' contains double quotes, which is not allowed"
        )
        # Add double quotes around the keyword
        return f'"{keyword}"'

    # Get the keywords lists and categories from dictionary
    keywords_lists = dictionary.get("keywords_lists", {})
    categories = dictionary.get("categories", {})

    # Process all query types in the categories
    for query_type, category_info in categories.items():
        # Get the query definition
        query_def = category_info.get("query", "")
        if not query_def:
            print(
                f"Warning: No query definition found for query type '{query_type}', skipping"
            )
            continue

        # Parse the query definition by splitting on '+'
        # Example: "war_keywords + threat_keywords"
        category_keys = [key.strip() for key in query_def.split("+")]

        # Group keywords by their categories
        grouped_keywords = []
        for category_key in category_keys:
            if category_key in keywords_lists:
                # Format each keyword with quotes
                formatted_keywords = [
                    format_keyword(keyword) for keyword in keywords_lists[category_key]
                ]
                # Join keywords within a category with "or"
                if len(formatted_keywords) > 1:
                    grouped_keywords.append(f"({' or '.join(formatted_keywords)})")
                elif len(formatted_keywords) == 1:
                    grouped_keywords.append(formatted_keywords[0])
            else:
                print(
                    f"Warning: Category '{category_key}' not found in keywords_lists for query type '{query_type}'"
                )

        # Join category groups with "and"
        if len(grouped_keywords) > 1:
            search_strings[query_type] = " and ".join(grouped_keywords)
        elif len(grouped_keywords) == 1:
            search_strings[query_type] = grouped_keywords[0]
        else:
            search_strings[query_type] = ""

    return search_strings
