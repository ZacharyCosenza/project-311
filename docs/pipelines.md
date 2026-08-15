cd /home/zaccosenza/code/project-311
1. raw — fetches calls/events/weather, no prerequisites.


.venv/bin/kedro run --pipeline raw
2. target — needs raw to have run first (reads calls_weekly).


.venv/bin/kedro run --pipeline target
3. features — needs raw + target.


.venv/bin/kedro run --pipeline features
4. modeling — needs features (trains, computes metrics, logs to MLflow, writes all the report plots including the new beeswarm).


.venv/bin/kedro run --pipeline modeling
__default__ — runs all four of the above in order in one go (this is what train actually is):


.venv/bin/kedro run
5. inference — needs model + modeling_data to already exist (i.e. modeling must have run at least once). Fetches its own narrow window fresh each time, produces inference_results.parquet.


.venv/bin/kedro run --pipeline inference
6. tweet_summary — needs inference_results (i.e. inference must have run). No day-of-week restriction, safe to run any day.


.venv/bin/kedro run --pipeline tweet_summary
7. tweet_daily — same prerequisite as tweet_summary, but only works Monday–Friday — select_daily_district deliberately raises on weekends (that's the guard that fired for you a few days ago). No workaround needed except running it on a weekday, or calling format_daily_deep_dive/plot_daily_trend directly in a Python shell if you want to test the logic on a weekend.


.venv/bin/kedro run --pipeline tweet_daily
Two things that apply to all of them:

Add KEDRO_ENV=prod before any command to target the data/prod/... paths instead of data/dev/... (matches what the deployed CronWorkflows use).
Without real Twitter credentials in your shell, tweet_summary/tweet_daily will run fully (including generating the map/trend images) but just print "no Twitter credentials set, not posting" instead of actually posting — safe to run freely for testing.