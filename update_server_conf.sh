#!/bin/bash

# Shell script to update server::kick_rand value in ~/.rcssserver/server.conf

CONFIG_FILE="$HOME/.rcssserver/server.conf"
BACKUP_FILE="$HOME/.rcssserver/backup_server.conf"

# Check if the backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file $BACKUP_FILE not found!"
    exit 1
fi

# Replace server.conf with backup_server.conf
cp "$BACKUP_FILE" "$CONFIG_FILE"
echo "Restored configuration from backup: $BACKUP_FILE -> $CONFIG_FILE"

# Function to replace a server configuration parameter
replace_server_param() {
    local param_name="$1"
    local new_value="$2"
    
    # Use sed to find and replace the line starting with the parameter
    # Using | as delimiter to handle paths with forward slashes
    sed -i "s|^server::${param_name} = .*|server::${param_name} = ${new_value}|" "$CONFIG_FILE"
    
    # Check if the replacement was successful
    if grep -q "^server::${param_name} = ${new_value}" "$CONFIG_FILE"; then
        echo "Successfully updated server::${param_name} to ${new_value}"
    else
        echo "Warning: server::${param_name} line may not have been found or updated"
    fi
}

replace_server_param "text_log_dir" "'./text_logs/'"
replace_server_param "game_log_dir" "'./game_logs/'"
replace_server_param "kick_off_wait" "10"
replace_server_param "kick_rand" "0.01"
replace_server_param "player_rand" "0.01"
replace_server_param "ball_rand" "0.01"
replace_server_param "fullstate_l" "true"
replace_server_param "fullstate_r" "true"

echo "Configuration file updated: $CONFIG_FILE"
