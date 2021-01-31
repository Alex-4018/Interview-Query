#1. Select the top 3 departments by the highest percentage of employees making over 100K in salary and have at least 10 employees.
SELECT CAST(SUM(
        CASE WHEN 
            salary > 100000 THEN 1 ELSE 0 
        END) AS DECIMAL
      )/COUNT(*) AS percentage_over_100K
      , d.name as department_name
      , COUNT(*) AS number_of_employees
FROM departments AS d
LEFT JOIN employees AS e
    ON d.id = e.department_id
GROUP BY d.name
HAVING COUNT(*) >= 10
ORDER BY 2 DESC
LIMIT 3


#2. Randomly sample a row from this table with over 100 millions records
SELECT r1.id, r1.name
FROM big_table AS r1 
INNER JOIN (
    SELECT CEIL(RAND() * (
        SELECT MAX(id)
        FROM big_table)
    ) AS id
) AS r2
    ON r1.id >= r2.id
ORDER BY r1.id ASC
LIMIT 1
#####################################
SELECT id, name 
FROM big_table
ORDER BY RAND()
LIMIT 1
####################################
SELECT * FROM big_table 
WHERE id >= CEIL(RAND() * ( SELECT MAX(id ) FROM big_table )) 
ORDER BY id ASC 
LIMIT 1

#3. Get the total three day rolling average for deposits by day
WITH bank AS 
(select DATE(created_at) AS dt, sum(transaction_value) as total 
from bank_transactions 
WHERE transaction_value >0 
group by DATE(created_at) )

select dt, avg(total) over(order by dt asc rows between 2 preceding and current row) as rolling_three_day 
from bank;

#4. Cumulative Distribution
with tmp as 
(select u.id, count(c.user_id) as num_comments 
 from users u left join comments c on u.id = c.user_id 
 group by 1),
tmp2 as 
(select num_comments, count(*) as freq 
 from tmp where num_comments > 0 
 group by 1 
 order by 1)

select t.num_comments, t.freq, sum(t2.freq) as cum_freq from tmp2 t left join tmp2 t2 on t.num_comments >= t2.num_comments group by 1
--select num_comments, freq, sum(freq) over (order by num_comments) as cum_freq from tmp2 --


























