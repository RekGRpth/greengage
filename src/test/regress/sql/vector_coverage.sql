-- ---- Vector coverage canary ----
-- Pins which plan shapes the vector post-pass takes for each expression
-- and operator form, under force.  Any change of the region eligibility
-- shows here as an expected-file diff, in both directions: a newly
-- wrapped form and a newly rejected one.  Keep the forms additive; when
-- a form's coverage intentionally changes, update the answer files with
-- the same discipline as any other expected output.
CREATE TABLE cov_aoco (a int, b bigint, c numeric(10,2), d text,
                       e bpchar(5), f float8, g date, h timestamp,
                       i bool, j int2, k varchar(10))
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (a);
INSERT INTO cov_aoco
SELECT i, i, (i % 100)::numeric(10,2), 'd' || (i % 7), 'E' || (i % 5),
       i * 1.5, date '2020-01-01' + (i % 30),
       timestamp '2020-01-01' + (i % 30) * interval '1 day',
       (i % 2) = 0, (i % 50)::int2, 'v' || (i % 9)
FROM generate_series(1, 2000) AS i;
CREATE TABLE cov_heap (a int, d text) DISTRIBUTED BY (a);
INSERT INTO cov_heap SELECT i, 'h' || (i % 11) FROM generate_series(1, 2000) AS i;
ANALYZE cov_aoco;
ANALYZE cov_heap;
SET gp_vectorized_execution = force;
-- comparisons, by specialized family
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE j = 3;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE a = 3;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE b = 3;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE c > 10;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE f > 1.5;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE g >= date '2020-01-10';
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE h >= timestamp '2020-01-10 00:00';
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE i;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE d = 'd3';
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE d <> 'd3';
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE e = 'E3';
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE k = 'v3';
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE d < 'd3';  -- ordering compare stays row-wise
-- boolean composition and conditional forms
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE a = 3 AND b = 3;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE a = 3 OR b = 3;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE d IS NULL OR e IS NOT NULL;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE i IS TRUE;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco
  WHERE CASE WHEN a % 2 = 0 THEN d ELSE e END = 'd3';
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE COALESCE(d, e) = 'd3';
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE NULLIF(a, 3) = 4;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE GREATEST(a, 3) = a;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE LEAST(a, 3) = 3;
-- arithmetic, by specialized family
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE a + 1 > 3;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE b * 2 > 3;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE c * 1.5 > 10;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE f * 2.0 > 1.5;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE a / 2 > 1;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE a % 2 = 0;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE a + a % 3 * 2 - 1 > 0;
-- IN lists: constant set (specialized) and dynamic array (generic)
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE a IN (1, 2, 3);
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE d IN ('d1', 'd2');
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE e NOT IN ('E1', 'E2');
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE a = ANY (ARRAY[1, a % 3]);
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE d LIKE 'd%';
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE d LIKE '%3' ESCAPE '#';
-- aggregates, plain and grouped
EXPLAIN (COSTS OFF) SELECT count(*), sum(a), avg(c), min(g), max(h) FROM cov_aoco;
EXPLAIN (COSTS OFF) SELECT a, count(*) FROM cov_aoco GROUP BY a;
EXPLAIN (COSTS OFF) SELECT d, count(*) FROM cov_aoco GROUP BY d;
EXPLAIN (COSTS OFF) SELECT count(DISTINCT a), count(DISTINCT d) FROM cov_aoco;
EXPLAIN (COSTS OFF) SELECT count(*) FILTER (WHERE i) FROM cov_aoco;
EXPLAIN (COSTS OFF) SELECT d, count(*) FROM cov_aoco GROUP BY d ORDER BY d LIMIT 5;
EXPLAIN (COSTS OFF) SELECT DISTINCT a FROM cov_aoco;
EXPLAIN (COSTS OFF) SELECT d, GROUPING(d) FROM cov_aoco GROUP BY ROLLUP (d);
-- joins
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco p JOIN cov_aoco q ON p.a = q.a;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco p JOIN cov_aoco q ON p.d = q.d;
SET enable_hashjoin = off;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco p JOIN cov_aoco q ON p.a = q.a;
RESET enable_hashjoin;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco p JOIN cov_heap q ON p.d = q.d;
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE EXISTS (SELECT 1 FROM cov_heap WHERE cov_heap.a = cov_aoco.a);
-- heap leaf through the row adapter, append over union all
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_heap WHERE d = 'h3';
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_heap WHERE a IN (1, 2);
EXPLAIN (COSTS OFF) SELECT count(*) FROM
  (SELECT a, d FROM cov_heap UNION ALL SELECT a, e FROM cov_aoco) t;
EXPLAIN (COSTS OFF) SELECT a, count(*) FROM
  (SELECT a, d FROM cov_heap UNION ALL SELECT a, d FROM cov_heap) t
GROUP BY a;
-- uncorrelated initplan/subplan
EXPLAIN (COSTS OFF) SELECT count(*) FROM cov_aoco WHERE a = (SELECT max(a) FROM cov_heap);
RESET gp_vectorized_execution;
DROP TABLE cov_aoco;
DROP TABLE cov_heap;
