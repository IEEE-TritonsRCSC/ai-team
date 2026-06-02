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
replace_server_param "kick_rand" "0.01"
replace_server_param "ball_rand" "0.01"
replace_server_param "coach_w_referee" "true"
replace_server_param "player_size" "0.9"
replace_server_param "ball_size" "0.215"
replace_server_param "kickable_margin" "0.1"
# Disable the BallStuckRef "drop ball" rule. Default is 100 cycles which
# fires inside our training episodes (model is at-ball but kicks don't
# move the ball reliably), teleporting our robot and corrupting transitions.
replace_server_param "drop_ball_time" "99999"
# Extend match length so the server doesn't enter `time_over` mid-training.
# Default half_time=300s × nr_normal_halfs=2 × 10 cycles/s = 6000 cycles (~7
# min wall-clock), after which no goals can be scored and reward = 0.
replace_server_param "half_time" "99999"
replace_server_param "nr_normal_halfs" "99999"
# Match the real robots' kick exit speed (~5.75 m/s, per mechanical-team video
# analysis). Scale: 1 sim unit = 0.1 m, 1 cycle = 0.1 s, so units/cycle = m/s.
# A power-100 kick imparts eff_power = 100*kick_power_rate at ideal contact,
# clipped by ball_accel_max then ball_speed_max. 100*0.0575 = 5.75 units/cycle,
# with accel/speed ceilings raised above it so they don't clip.
# NOTE: ball_decay (friction) left at default 0.94 for now — refine once the
# mechanical team provides velocity-loss-per-meter (matters for Stage 2+).
replace_server_param "kick_power_rate" "0.0575"
replace_server_param "ball_accel_max" "5.8"
replace_server_param "ball_speed_max" "6"

echo "Configuration file updated: $CONFIG_FILE"
