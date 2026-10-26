-- Example queries. Run all:  python -m sei.sql --file queries/examples.sql
-- Or paste one into:         python -m sei.sql
--
-- users            one row per user (arm, age_group, region, device, account_type)
-- daily_activity   one row per active user-day (day -14..27)
-- user_experiment  one row per user with pre_*, exp_* and late_* metrics and churned
-- churn_scores     lapse probability for users active on day 13
-- user_cate        predicted treatment effects for held-out users

-- 1. How many users are in each arm?
SELECT arm, COUNT(*) AS users, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM users GROUP BY arm ORDER BY arm;

-- 2. Average daily minutes per user, before vs during the experiment
SELECT arm,
       ROUND(AVG(pre_minutes_per_day), 2) AS pre_minutes,
       ROUND(AVG(exp_minutes_per_day), 2) AS exp_minutes,
       ROUND(100 * (AVG(exp_minutes_per_day) / AVG(pre_minutes_per_day) - 1), 1) AS change_pct
FROM user_experiment GROUP BY arm ORDER BY arm;

-- 3. Raw lift vs control (no CUPED)
WITH a AS (
  SELECT arm, AVG(exp_minutes_per_day) AS minutes, AVG(churned) AS churn,
         AVG(exp_friend_interactions_per_day) AS friend_int, AVG(exp_ad_revenue_per_day) AS revenue
  FROM user_experiment GROUP BY arm)
SELECT t.arm,
       ROUND(100 * (t.minutes / c.minutes - 1), 1)       AS minutes_lift_pct,
       ROUND(100 * (t.churn - c.churn), 2)               AS churn_change_pp,
       ROUND(100 * (t.friend_int / c.friend_int - 1), 1) AS friend_interactions_lift_pct,
       ROUND(100 * (t.revenue / c.revenue - 1), 1)       AS revenue_lift_pct
FROM a t CROSS JOIN (SELECT * FROM a WHERE arm = 'control') c
WHERE t.arm <> 'control';

-- 4. Churn rate by age group and arm
PIVOT (SELECT age_group, arm, churned FROM user_experiment)
ON arm USING ROUND(100 * AVG(churned), 2)
GROUP BY age_group ORDER BY age_group;

-- 5. Weekly minutes per assigned user, by arm
SELECT d.day // 7 + 1 AS week, u.arm,
       ROUND(SUM(d.minutes) / (7.0 * ANY_VALUE(n.users)), 2) AS minutes_per_user_day
FROM daily_activity d
JOIN users u USING (user_id)
JOIN (SELECT arm, COUNT(*) AS users FROM users GROUP BY arm) n ON n.arm = u.arm
WHERE d.day >= 0
GROUP BY week, u.arm ORDER BY week, u.arm;

-- 6. Daily active users (DAU) over time
SELECT d.date, u.arm, COUNT(*) AS active_users
FROM daily_activity d JOIN users u USING (user_id)
GROUP BY d.date, u.arm ORDER BY d.date, u.arm LIMIT 15;

-- 7. Which regions earn the most ad revenue per user?
SELECT region, COUNT(*) AS users, ROUND(AVG(exp_ad_revenue_per_day) * 365, 2) AS annual_revenue_per_user
FROM user_experiment GROUP BY region ORDER BY annual_revenue_per_user DESC;

-- 8. Creators vs casual users: posting under each arm
SELECT account_type, arm, ROUND(AVG(exp_posts_created_per_day), 3) AS posts_per_day
FROM user_experiment GROUP BY account_type, arm ORDER BY account_type, arm;

-- 9. Users most at risk of a 7-day lapse
SELECT user_id, arm, age_group, region, ROUND(churn_probability, 3) AS lapse_probability, risk_tier
FROM churn_scores ORDER BY churn_probability DESC LIMIT 10;

-- 10. Predicted vs actual lapse rate by risk decile
SELECT risk_decile, COUNT(*) AS users, ROUND(100 * AVG(churn_probability), 1) AS predicted_pct,
       ROUND(100 * AVG(actual_churn_next_14d), 1) AS actual_pct
FROM churn_scores GROUP BY risk_decile ORDER BY risk_decile DESC;

-- 11. Arm picked by the personalisation model, by age group
SELECT age_group, policy_arm, COUNT(*) AS users
FROM user_cate GROUP BY age_group, policy_arm ORDER BY age_group, users DESC;

-- 12. Guardrail results (CUPED, BH-adjusted)
SELECT arm, label, ROUND(100 * rel_lift, 1) AS lift_pct, p_adj_bh, verdict
FROM metric_results WHERE tier = 'guardrail' ORDER BY arm, label;

-- 13. Segments where the AI feed significantly raises churn
SELECT dimension, segment, ROUND(churn_diff_pp, 2) AS churn_change_pp, ROUND(churn_p_adj, 4) AS p_adj
FROM segment_results WHERE arm = 'ai_feed' AND status = 'churn risk' ORDER BY churn_change_pp DESC;

-- 14. Viral rate by news channel (real data)
SELECT level AS channel, articles, median_shares, ROUND(100 * viral_rate, 1) AS viral_pct
FROM virality_playbook WHERE dimension = 'channel' ORDER BY viral_pct DESC
