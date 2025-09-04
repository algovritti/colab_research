import pandas as pd

def backtest_intraday_open_breakout(hist_df, pct=0.005, sl_pct=None, start_time="0:15", end_time="23:35"):
    """
    hist_df: 5-min dataframe with columns: open, high, low, close, time
    pct: decimal fraction for trigger (0.005 = 0.5%)
    sl_pct: decimal fraction for stop (if None, uses same as pct)
    Returns: trades_df, metrics_dict
    """
    # normalize stoploss
    if sl_pct is None:
        sl_pct = pct

    # ensure datetime index
    df = hist_df.copy()
    df.columns = df.columns.str.lower()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    trades = []

    # add simple date column for grouping
    df["trade_date"] = df.index.date

    for day, day_data in df.groupby("trade_date"):
        # restrict to trading hours
        day_data = day_data.between_time(start_time, end_time)
        if day_data.empty:
            continue

        # daily open = first candle open
        day_open = float(day_data.iloc[0]["open"])
        long_trigger = day_open * (1 + pct)
        short_trigger = day_open * (1 - pct)

        entry, trigger_ts, trade_type, sl = None, None, None, None

        # find first trigger
        for ts, row in day_data.iterrows():
            if row["high"] >= long_trigger:
                entry = long_trigger
                trigger_ts = ts
                trade_type = "LONG"
                sl = entry * (1 - sl_pct)
                break
            elif row["low"] <= short_trigger:
                entry = short_trigger
                trigger_ts = ts
                trade_type = "SHORT"
                sl = entry * (1 + sl_pct)
                break

        if entry is None:
            continue

        # check stoploss
        after = day_data.loc[trigger_ts:].iloc[1:]
        exit_price, exit_ts, stop_hit = None, None, False

        for ts, row in after.iterrows():
            if trade_type == "LONG":
                if float(row["low"]) <= sl:
                    exit_price, exit_ts, stop_hit = sl, ts, True
                    break
            else:
                if float(row["high"]) >= sl:
                    exit_price, exit_ts, stop_hit = sl, ts, True
                    break

        if exit_price is None:
            exit_price = float(day_data.iloc[-1]["close"])
            exit_ts = day_data.index[-1]

        pnl = (exit_price - entry) if trade_type == "LONG" else (entry - exit_price)

        trades.append({
            "Date": pd.to_datetime(str(day)),
            "Type": trade_type,
            "Entry": entry,
            "Exit": exit_price,
            "PnL": pnl,
            "StopHit": bool(stop_hit),
            "EntryTime": trigger_ts,
            "ExitTime": exit_ts
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        metrics = {"Total_PnL": 0.0, "Max_Drawdown": 0.0, "Total_Trades": 0, "Win_Rate_pct": 0.0, "Avg_PnL": 0.0}
        return trades_df, metrics

    # metrics
    trades_df = trades_df.sort_values("Date").reset_index(drop=True)
    total_pnl = trades_df["PnL"].sum()
    cumulative = trades_df["PnL"].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    total_trades = len(trades_df)
    win_rate = (trades_df["PnL"] > 0).mean() * 100
    avg_pnl = trades_df["PnL"].mean()

    metrics = {
        "Total_PnL": float(total_pnl),
        "Max_Drawdown": float(max_dd),
        "Total_Trades": int(total_trades),
        "Win_Rate_pct": float(win_rate),
        "Avg_PnL": float(avg_pnl)
    }

    return trades_df, metrics



def backtest_intraday_open_breakout2(hist_df, pct=0.005, sl_pct=None, start_time="00:15", end_time="23:35"):
    """
    hist_df: 5-min dataframe with DatetimeIndex and columns: open, high, low, close (case-insensitive)
    pct: decimal fraction for trigger (0.005 = 0.5%)
    sl_pct: decimal fraction for stop (if None, uses same as pct)
    Returns: trades_df, metrics_dict
    """
    if sl_pct is None:
        sl_pct = pct
    df = hist_df.copy()
    df.columns = df.columns.str.lower()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    trades = []
    df["trade_date"] = df.index.date

    for day, day_data in df.groupby("trade_date"):
        day_data = day_data.between_time(start_time, end_time)
        if day_data.empty:
            continue

        # daily open
        day_open = float(day_data.iloc[0]["open"])
        long_trigger = day_open * (1 + pct)
        short_trigger = day_open * (1 - pct)

        trade_count = 0
        entry, trigger_ts, trade_type, sl = None, None, None, None

        # --- First Trade ---
        for ts, row in day_data.iterrows():
            if row["high"] >= long_trigger:
                entry, trigger_ts, trade_type = long_trigger, ts, "LONG"
                sl = entry * (1 - sl_pct)
                break
            elif row["low"] <= short_trigger:
                entry, trigger_ts, trade_type = short_trigger, ts, "SHORT"
                sl = entry * (1 + sl_pct)
                break

        if entry is None:
            continue

        # Manage first trade
        after = day_data.loc[trigger_ts:].iloc[1:]
        exit_price, exit_ts, stop_hit = None, None, False

        for ts, row in after.iterrows():
            if trade_type == "LONG":
                if float(row["low"]) <= sl:  # SL hit
                    exit_price, exit_ts, stop_hit = sl, ts, True
                    break
            else:  # SHORT
                if float(row["high"]) >= sl:
                    exit_price, exit_ts, stop_hit = sl, ts, True
                    break

        if exit_price is None:  # SL not hit
            exit_price, exit_ts, stop_hit = float(day_data.iloc[-1]["close"]), day_data.index[-1], False

        pnl = (exit_price - entry) if trade_type == "LONG" else (entry - exit_price)
        trades.append({"Date": pd.to_datetime(str(day)), "Type": trade_type,
                       "Entry": entry, "Exit": exit_price, "PnL": pnl,
                       "StopHit": stop_hit, "EntryTime": trigger_ts, "ExitTime": exit_ts})
        trade_count += 1

        # --- Second Trade (only if SL hit and trade_count < 2) ---
        if stop_hit and trade_count < 2:
            opposite_type = "SHORT" if trade_type == "LONG" else "LONG"
            opp_entry = exit_price  # enter at SL price
            opp_trigger_ts = exit_ts
            opp_sl = opp_entry * (1 - sl_pct) if opposite_type == "LONG" else opp_entry * (1 + sl_pct)

            after2 = day_data.loc[opp_trigger_ts:].iloc[1:]
            opp_exit_price, opp_exit_ts, opp_stop_hit = None, None, False

            for ts, row in after2.iterrows():
                if opposite_type == "LONG":
                    if float(row["low"]) <= opp_sl:
                        opp_exit_price, opp_exit_ts, opp_stop_hit = opp_sl, ts, True
                        break
                else:  # SHORT
                    if float(row["high"]) >= opp_sl:
                        opp_exit_price, opp_exit_ts, opp_stop_hit = opp_sl, ts, True
                        break

            if opp_exit_price is None:
                opp_exit_price, opp_exit_ts, opp_stop_hit = float(day_data.iloc[-1]["close"]), day_data.index[-1], False

            pnl2 = (opp_exit_price - opp_entry) if opposite_type == "LONG" else (opp_entry - opp_exit_price)
            trades.append({"Date": pd.to_datetime(str(day)), "Type": opposite_type,
                           "Entry": opp_entry, "Exit": opp_exit_price, "PnL": pnl2,
                           "StopHit": opp_stop_hit, "EntryTime": opp_trigger_ts, "ExitTime": opp_exit_ts})
            trade_count += 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        metrics = {"Total_PnL": 0.0, "Max_Drawdown": 0.0, "Total_Trades": 0, "Win_Rate_pct": 0.0, "Avg_PnL": 0.0}
        return trades_df, metrics

    # metrics
    trades_df = trades_df.sort_values(["Date", "EntryTime"]).reset_index(drop=True)
    total_pnl = trades_df["PnL"].sum()
    cumulative = trades_df["PnL"].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    total_trades = len(trades_df)
    win_rate = (trades_df["PnL"] > 0).mean() * 100
    avg_pnl = trades_df["PnL"].mean()

    metrics = {"Total_PnL": float(total_pnl),
               "Max_Drawdown": float(max_dd),
               "Total_Trades": int(total_trades),
               "Win_Rate_pct": float(win_rate),
               "Avg_PnL": float(avg_pnl)}

    return trades_df, metrics



btc_hist_data = pd.read_excel("btc_intraday_candles_2024-2025.xlsx")

# Ensure 'time' is datetime
btc_hist_data['time'] = pd.to_datetime(btc_hist_data['time'])

# Function to filter by date range
def filter_by_date(df, start_date, end_date):
    """
    Returns a new DataFrame with records between start_date and end_date.
    
    Args:
        df (pd.DataFrame): Original DataFrame with a 'time' column.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
    """
    mask = (df['time'] >= start_date) & (df['time'] <= end_date)
    return df.loc[mask].reset_index(drop=True)

# Example usage
filtered_df = filter_by_date(btc_hist_data, "2024-01-01", "2025-01-01")

trades, metrics = backtest_intraday_open_breakout2(filtered_df, pct=0.003, sl_pct=0.002, start_time="0:50", end_time="23:35")
print(metrics)
#print(trades.head(50))
#trades.to_excel("btc_open_breakout3_sl2_2025.xlsx")


trades, metrics = backtest_intraday_open_breakout(filtered_df, pct=0.004, sl_pct=0.004, start_time="0:10", end_time="23:15")
print(metrics)