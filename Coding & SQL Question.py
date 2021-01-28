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



