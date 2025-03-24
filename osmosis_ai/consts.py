from enum import Enum

# Extract package metadata
package_name = "osmosis-ai"
package_version = "0.1.6"

indent = 2  # Number of spaces to use for indentation in pretty print
osmosis_api_url = "https://ftgrv77m9f.execute-api.us-west-2.amazonaws.com/prod"

DEFAULT_LOG_DESTINATION = "none"  # "none" or "stdout" or "stderr" or "file"

class LogDestination(Enum):
    NONE = "none"
    STDOUT = "stdout"
    STDERR = "stderr"
    FILE = "file"