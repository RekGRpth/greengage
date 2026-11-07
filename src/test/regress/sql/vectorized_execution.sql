SET client_min_messages = warning;
DROP TABLE IF EXISTS vector_aoco_test;
DROP TABLE IF EXISTS vector_aoco_small;
DROP TABLE IF EXISTS vector_join_l;
DROP TABLE IF EXISTS vector_join_r;
DROP TABLE IF EXISTS vector_aoco_wide;
DROP TABLE IF EXISTS vector_aoco_spill;
DROP TABLE IF EXISTS vector_aoco_spill_wide;
DROP TABLE IF EXISTS vector_heap_test;
DROP TABLE IF EXISTS vector_aoro_test;
DROP TABLE IF EXISTS vector_limit_test;
DROP TABLE IF EXISTS vector_unique_test;
CREATE TABLE vector_aoco_test
(id int4, big int8, val text, num numeric(12,2))
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (id);
INSERT INTO vector_aoco_test
SELECT g, g::int8 * 10,
       CASE WHEN g % 7 = 0 THEN NULL ELSE 'v' || g END,
       (g % 100)::numeric / 3
FROM generate_series(1, 5000) AS g;
ALTER TABLE vector_aoco_test ADD COLUMN later int4 DEFAULT 42;
INSERT INTO vector_aoco_test
SELECT 5000 + g, (5000 + g)::int8 * 10, 'new' || g, 1.25, g
FROM generate_series(1, 10) AS g;
DELETE FROM vector_aoco_test WHERE id % 11 = 0;
ANALYZE vector_aoco_test;
CREATE TABLE vector_aoco_small (id int4, val text)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (id);
INSERT INTO vector_aoco_small
SELECT g, 'v' || g FROM generate_series(1, 10) AS g;
ANALYZE vector_aoco_small;
CREATE TABLE vector_limit_test (id int4, val text)
WITH (appendonly = true, orientation = column) DISTRIBUTED REPLICATED;
INSERT INTO vector_limit_test
SELECT g, 'v' || g FROM generate_series(1, 10) AS g;
ANALYZE vector_limit_test;
CREATE TABLE vector_unique_test (id int4, grp int4, val text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (grp);
INSERT INTO vector_unique_test
SELECT g, g % 100, 'v' || g FROM generate_series(1, 10000) AS g;
ANALYZE vector_unique_test;
CREATE TABLE vector_heap_test (id int4, big int8, val text, num numeric(12,2))
DISTRIBUTED BY (id);
CREATE TABLE vector_aoro_test (id int4, big int8, val text, num numeric(12,2))
WITH (appendonly = true, orientation = row) DISTRIBUTED BY (id);
INSERT INTO vector_heap_test
SELECT g, g::int8 * 10,
       CASE WHEN g % 7 = 0 THEN NULL ELSE 'v' || g END,
       (g % 100)::numeric / 3
FROM generate_series(1, 10000) AS g;
INSERT INTO vector_aoro_test SELECT * FROM vector_heap_test;
ANALYZE vector_heap_test;
ANALYZE vector_aoro_test;
CREATE TABLE vector_join_l (id int4, l text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (id);
CREATE TABLE vector_join_r (id int4, r text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (id);
INSERT INTO vector_join_l SELECT g, 'l' || g FROM generate_series(1, 5) AS g;
INSERT INTO vector_join_r VALUES (3, 'r3a'), (3, 'r3b'), (4, 'r4'), (6, 'r6'), (8, 'r8');
ANALYZE vector_join_l;
ANALYZE vector_join_r;
CREATE TABLE vector_aoco_wide (id int4, pad text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (id);
INSERT INTO vector_aoco_wide SELECT g, repeat('x', 1000) FROM generate_series(1, 5000) AS g;
ANALYZE vector_aoco_wide;
CREATE TABLE vector_aoco_spill (grp int4, pad text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (grp);
INSERT INTO vector_aoco_spill SELECT g % 10, repeat('x', 1000) FROM generate_series(1, 5000) AS g;
ANALYZE vector_aoco_spill;
CREATE TABLE vector_aoco_spill_wide (id int4, pad text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (id);
INSERT INTO vector_aoco_spill_wide SELECT g, repeat('x', 1000) FROM generate_series(1, 50000) AS g;
ANALYZE vector_aoco_spill_wide;
SET gp_vectorized_execution = force;
EXPLAIN (COSTS OFF)
SELECT id AS c1, big AS c2, val AS c3, num AS c4, later AS c5
FROM vector_aoco_test ORDER BY id LIMIT 3;
EXPLAIN (COSTS OFF) SELECT id, val FROM vector_limit_test LIMIT 3;
SELECT id, val FROM vector_limit_test LIMIT 3;
SELECT id FROM vector_limit_test LIMIT 0;
SELECT id FROM vector_limit_test OFFSET 8;
SELECT id FROM vector_limit_test LIMIT 3 OFFSET 8;
SELECT id FROM vector_limit_test LIMIT NULL OFFSET NULL;
SELECT id FROM vector_limit_test LIMIT 3 OFFSET 100;
SELECT id FROM vector_limit_test LIMIT -1;
SELECT id FROM vector_limit_test OFFSET -1;
SELECT sum(big), sum(id), min(num), max(num), min(later), max(later)
FROM vector_aoco_test;
SELECT id, big, val, num, later
FROM vector_aoco_test
ORDER BY id OFFSET 2490 LIMIT 3;
SELECT val, count(*), sum(big)
FROM vector_aoco_test
GROUP BY val ORDER BY val LIMIT 3;
SELECT count(*), sum(a.id), sum(b.id),
       sum(length(a.val)), sum(length(b.val))
FROM vector_aoco_small a CROSS JOIN vector_aoco_small b;
SELECT id, big, val, num, later
FROM vector_aoco_test
WHERE id % 3 = 0 AND big > 25000 AND (val IS NULL OR num < 20.0)
ORDER BY id LIMIT 5;
SELECT sum(id) FROM vector_aoco_small
WHERE id < 0 AND 1 / (id - id) = 1;
SELECT sum(id) FROM vector_aoco_small
WHERE id >= 0 OR 1 / (id - id) = 1;
SELECT sum(id) FROM vector_aoco_small WHERE NOT (id < 5);
SELECT sum(id) FROM vector_aoco_small WHERE (val || '') = 'v' || id;
SELECT sum(id) FROM vector_aoco_test WHERE length(val) > 0;
SELECT count(*) FROM vector_aoco_test WHERE id IN (1, 2, 3, 5000, 99999);
SELECT count(*) FROM vector_aoco_test WHERE id NOT IN (4, 5, 6000);
SELECT count(*) FROM vector_aoco_test WHERE val IN ('new1', 'new2', 'v10', 'nope');
SELECT count(*) FROM vector_aoco_test WHERE num IN (1.25, 0.33);
SELECT id, ARRAY[id, NULL, id + 1] AS a FROM vector_aoco_small WHERE id < 4 ORDER BY id;
SELECT id, id = ANY(ARRAY[id % 5, 2]) AS any_match FROM vector_aoco_small ORDER BY id;
SELECT count(*) FROM vector_aoco_small WHERE id = ANY(ARRAY[]::int[]);
SELECT count(*) FROM vector_aoco_small WHERE id = ANY(NULL::int[]);
SELECT count(*) FROM vector_aoco_small WHERE id <> ALL(ARRAY[id % 5, 2]);
SELECT id, (ARRAY[id, id + 1, id + 2])[2] AS second FROM vector_aoco_small WHERE id < 4 ORDER BY id;
SELECT id, (ARRAY[id, id + 1, id + 2])[id] AS elem FROM vector_aoco_small WHERE id <= 3 ORDER BY id;
SELECT (ARRAY[1, 2, 3])[0] AS below, (ARRAY[1, 2, 3])[NULL] AS null_index,
       (NULL::int[])[1] AS null_array;
SELECT (ARRAY[1, 2, 3, 4])[2:3] AS mid, (ARRAY[1, 2, 3, 4])[2:] AS tail,
       (ARRAY[1, 2, 3, 4])[:2] AS head;
SELECT id, (ARRAY[1, 2, 3, 4])[id:3] AS part FROM vector_aoco_small WHERE id < 4 ORDER BY id;
SELECT (ARRAY[1, 2, 3, 4])[NULL:2] AS null_lower, (ARRAY[1, 2, 3, 4])[2:NULL] AS null_upper;
SELECT (ARRAY[1, 2, 3, 4])[1:2][1] AS nested;
SELECT ARRAY[1, 2, NULL]::text[] AS a;
SELECT ARRAY['1', '2', '3']::int[] AS a;
SELECT ARRAY[1.25, 2.5]::int[] AS a;
SELECT ARRAY[1, 2, 3]::oid[] AS a;
SELECT NULL::int[]::text[] AS a;
SELECT (ARRAY[1, 2, 3, 4])[1:2]::text[] AS a;
CREATE TYPE vector_pair AS (a int4, b text);
SELECT ROW(1, 'x') AS anon_pair, ROW(1, 'x')::vector_pair AS named_pair;
SELECT (ROW(id, 'v' || id)::vector_pair).a AS a,
       (ROW(id, 'v' || id)::vector_pair).b AS b
FROM vector_aoco_small WHERE id < 4 ORDER BY id;
SELECT (ROW(1, NULL)::vector_pair).b AS null_field;
SELECT (NULL::vector_pair).a AS null_record;
SELECT sum((ROW(id, id + 1)::vector_pair).a) FROM vector_aoco_small;
DROP TYPE vector_pair;
CREATE TABLE vector_parent (a int4, b text);
CREATE TABLE vector_child (c int4) INHERITS (vector_parent);
SELECT (ROW(1, 'x', 5)::vector_child)::vector_parent AS converted;
SELECT ((ROW(1, 'x', 5)::vector_child)::vector_parent).a AS a,
       ((ROW(1, 'x', 5)::vector_child)::vector_parent).b AS b;
SELECT (NULL::vector_child)::vector_parent AS null_conversion;
SELECT sum(((ROW(id, 'v' || id, id + 1)::vector_child)::vector_parent).a)
FROM vector_aoco_small;
DROP TABLE vector_child, vector_parent;
SELECT vector_aoco_small FROM vector_aoco_small ORDER BY id LIMIT 1;
SELECT (vector_aoco_small).id AS id, (vector_aoco_small).val AS val
FROM vector_aoco_small WHERE id < 4 ORDER BY id;
SELECT (vector_heap_test).id AS id FROM vector_heap_test WHERE id < 4 ORDER BY id;
SELECT sum((vector_aoco_small).id) FROM vector_aoco_small;
CREATE TABLE vector_whole_test (id int4, val text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (id);
INSERT INTO vector_whole_test VALUES (1, 'x'), (2, 'y');
ALTER TABLE vector_whole_test DROP COLUMN val;
SELECT vector_whole_test FROM vector_whole_test ORDER BY id;
DROP TABLE vector_whole_test;
SELECT current_user = session_user AS same_user,
       current_database() = current_catalog AS same_db,
       current_schema() = current_schema AS same_schema,
       current_role = current_user AS same_role;
SELECT current_date = current_date AS d, current_time = current_time AS t,
       current_timestamp = current_timestamp AS ts,
       localtime = localtime AS lt, localtimestamp = localtimestamp AS lts;
SELECT current_time(2) = current_time(2) AS t,
       current_timestamp(2) = current_timestamp(2) AS ts,
       localtime(2) = localtime(2) AS lt,
       localtimestamp(2) = localtimestamp(2) AS lts;
SELECT count(*) FROM vector_aoco_small WHERE current_schema IS NOT NULL;
CREATE SEQUENCE vector_seq;
SELECT nextval('vector_seq') FROM vector_limit_test LIMIT 3;
SELECT nextval('vector_seq') FROM vector_limit_test LIMIT 3;
DROP SEQUENCE vector_seq;
SELECT sum(nullif(id % 3, 0)), count(nullif(val, 'v1')), count(nullif(val, NULL))
FROM vector_aoco_test;
SELECT id, nullif(id % 3, 0) AS nz FROM vector_aoco_small WHERE id < 7 ORDER BY id;
SELECT count(*) FROM vector_aoco_test WHERE nullif(id % 3, 0) IS NULL;
SELECT count(*) FROM vector_aoco_small WHERE id IS DISTINCT FROM 3;
SELECT count(*) FROM vector_aoco_small WHERE id IS NOT DISTINCT FROM 3;
SELECT count(*) FROM vector_aoco_test WHERE val IS DISTINCT FROM 'new1';
SELECT count(*) FROM vector_aoco_test WHERE val IS NOT DISTINCT FROM NULL;
SELECT id, greatest(id % 5, 2) AS g, least(id % 5, 2) AS l
FROM vector_aoco_small ORDER BY id;
SELECT sum(greatest(id % 3, 1)), sum(least(id % 3, 1)) FROM vector_aoco_test;
SELECT count(greatest(val, 'fallback')), count(least(val, 'fallback')),
       count(greatest(val, NULL)), count(least(val, NULL))
FROM vector_aoco_test;
SELECT sum(greatest(num, 10)), sum(least(num, 10)) FROM vector_aoco_test;
SELECT coalesce(val, 'MISSING'), count(*)
FROM vector_aoco_test GROUP BY 1 ORDER BY 1 LIMIT 4;
SELECT sum(coalesce(later, 0)), count(coalesce(val, 'x')) FROM vector_aoco_test;
SELECT sum(CASE WHEN id % 2 = 0 THEN big ELSE -big END) FROM vector_aoco_test;
SELECT CASE WHEN num > 20 THEN 'high' WHEN num > 5 THEN 'mid' ELSE 'low' END AS bucket, count(*)
FROM vector_aoco_test GROUP BY 1 ORDER BY 1;
SELECT count(*) FROM vector_aoco_test
WHERE CASE WHEN val IS NULL THEN false ELSE length(val) > 4 END;
SELECT count(*), sum(CASE WHEN val IS NULL THEN 1 ELSE 0 END) FROM vector_aoco_test;
SELECT count(*) FROM vector_aoco_test WHERE (id % 2 = 0) IS TRUE;
SELECT count(*) FROM vector_aoco_test WHERE (id % 11 = 0) IS NOT TRUE;
SELECT count(*) FROM vector_aoco_test WHERE (val IS NULL) IS UNKNOWN;
SELECT count(*) FROM vector_aoco_test WHERE (val IS NULL) IS NOT UNKNOWN;
SELECT sum(CASE WHEN id % 3 = 0 THEN num * 2 WHEN id % 3 = 1 THEN num ELSE NULL END)
FROM vector_aoco_test;
SELECT CASE id % 4 WHEN 0 THEN 'zero' WHEN 1 THEN 'one' ELSE 'other' END AS bucket,
       count(*)
FROM vector_aoco_test GROUP BY 1 ORDER BY 1;
SELECT sum(CASE id % 3 WHEN 0 THEN big WHEN 1 THEN -big ELSE 0 END)
FROM vector_aoco_test;
SELECT count(*) FROM vector_aoco_test
WHERE CASE val WHEN 'new1' THEN true WHEN NULL THEN true ELSE false END;
SELECT sum(CASE id % 2 WHEN 0 THEN CASE WHEN num > 10 THEN 1 ELSE 2 END ELSE 0 END)
FROM vector_aoco_test;
SELECT sum(CASE id % 2 WHEN 0 THEN CASE id % 3 WHEN 0 THEN 1 ELSE 2 END ELSE 0 END)
FROM vector_aoco_test;
SELECT sum(id) FROM vector_aoco_small
WHERE CASE id WHEN 0 THEN 1 / (id - id) = 1 ELSE true END;
SELECT count(CASE id % 10 WHEN 0 THEN id END) FROM vector_aoco_test;
SELECT id, big + 1 AS b1 FROM vector_aoco_test ORDER BY id LIMIT 5;
SELECT id, CASE id % 2 WHEN 0 THEN 'even' ELSE 'odd' END AS parity
FROM vector_aoco_test WHERE id BETWEEN 4998 AND 5010 ORDER BY id;
SELECT id, coalesce(val, 'NONE') FROM vector_aoco_test WHERE id % 500 = 0 ORDER BY id;
SELECT id FROM vector_aoco_test ORDER BY big DESC LIMIT 3;
EXPLAIN (COSTS OFF) SELECT id, big FROM vector_aoco_test ORDER BY id DESC LIMIT 5;
SELECT id, big FROM vector_aoco_test ORDER BY id DESC LIMIT 5;
SELECT id, val FROM vector_aoco_test WHERE id % 500 = 0
ORDER BY val NULLS FIRST, id;
EXPLAIN (COSTS OFF)
SELECT DISTINCT ON (grp) id, val FROM vector_unique_test
WHERE id <= 20 ORDER BY grp, id;
SELECT DISTINCT ON (grp) id, val FROM vector_unique_test
WHERE id <= 20 ORDER BY grp, id;
SELECT DISTINCT ON (grp) id FROM vector_unique_test
ORDER BY grp, id OFFSET 1 LIMIT 1;
SELECT DISTINCT ON (grp) id FROM vector_unique_test ORDER BY grp, id LIMIT 3;
SELECT id, (SELECT max(big) FROM vector_aoco_small s WHERE s.id = t.id) AS m
FROM vector_aoco_test t WHERE t.id < 12 ORDER BY id;
SELECT t.id, EXISTS (SELECT 1 FROM vector_aoco_small s WHERE s.id = t.id) AS has_s
FROM vector_aoco_test t WHERE t.id < 8 ORDER BY t.id;
EXPLAIN (COSTS OFF)
SELECT count(*) FROM vector_aoco_test l
JOIN (SELECT id % 10 AS g, count(*) AS c FROM vector_aoco_small GROUP BY 1) r
  ON l.id % 10 = r.g;
SELECT count(*) FROM vector_aoco_test l
JOIN (SELECT id % 10 AS g, count(*) AS c FROM vector_aoco_small GROUP BY 1) r
  ON l.id % 10 = r.g;
SELECT count(*) FROM vector_aoco_test l
LEFT JOIN (SELECT id % 3 AS g FROM vector_aoco_small GROUP BY 1) r
  ON l.id % 3 = r.g;
SELECT count(*) FROM vector_aoco_test;
SELECT count(*), sum(id) FROM vector_aoco_small WHERE id < 0;
SELECT val, count(*) FROM vector_aoco_small WHERE id < 0 GROUP BY val;
SELECT later, val, count(*), sum(big), min(num), max(num), min(val), max(val)
FROM vector_aoco_test
GROUP BY later, val ORDER BY later, val LIMIT 5;
SELECT count(*) FILTER (WHERE id % 2 = 0),
       sum(id) FILTER (WHERE id % 3 = 0),
       count(val) FILTER (WHERE num > 10)
FROM vector_aoco_test;
SELECT sum(1 / (id - id)) FILTER (WHERE id < 0) FROM vector_aoco_small;
SELECT id % 3 AS g, count(*) FILTER (WHERE val IS NULL),
       sum(id) FILTER (WHERE id % 2 = 0)
FROM vector_aoco_test GROUP BY 1 ORDER BY 1;
SELECT count(*) FROM vector_aoco_test HAVING count(*) > 4000;
SELECT count(*) FROM vector_aoco_small WHERE id < 0 HAVING count(*) > 0;
SELECT id % 3 AS g, count(*) FROM vector_aoco_test
GROUP BY 1 HAVING count(*) > 1500 AND sum(id) > 0 ORDER BY 1;
SELECT id % 4 AS g, count(*) FILTER (WHERE val IS NULL) AS nulls
FROM vector_aoco_test GROUP BY 1 HAVING count(*) FILTER (WHERE val IS NULL) > 100
ORDER BY 1;
SELECT val, count(*) FROM vector_aoco_small GROUP BY val HAVING count(*) = 0;
EXPLAIN (COSTS OFF) SELECT count(DISTINCT id) FROM vector_aoco_small;
SELECT count(DISTINCT id), sum(DISTINCT id % 100),
       count(DISTINCT val), sum(DISTINCT num)
FROM vector_aoco_test;
SELECT count(DISTINCT id) FILTER (WHERE id % 2 = 0) FROM vector_aoco_test;
SELECT count(DISTINCT id) FROM vector_aoco_small WHERE id < 0;
EXPLAIN (COSTS OFF)
SELECT string_agg(val, ',' ORDER BY id DESC) FROM vector_aoco_small;
SELECT string_agg(val, ',' ORDER BY id) FROM vector_aoco_small;
SELECT string_agg(val, ',' ORDER BY id DESC) FROM vector_aoco_small;
SELECT array_agg(id ORDER BY id DESC) FROM vector_aoco_small WHERE id <= 5;
SELECT array_agg(id ORDER BY id) FILTER (WHERE id % 2 = 0) FROM vector_aoco_small;
SELECT array_agg(DISTINCT id % 3 ORDER BY id % 3) FROM vector_aoco_small;
SELECT array_agg(id ORDER BY id) FROM vector_aoco_small WHERE id < 0;
SELECT string_agg(val || ':' || id, '|' ORDER BY num DESC, id)
FROM vector_aoco_test WHERE id < 10;
EXPLAIN (COSTS OFF)
SELECT id % 3 AS g, string_agg(val, ',' ORDER BY id)
FROM vector_aoco_small GROUP BY 1 ORDER BY 1;
SELECT id % 3 AS g, count(DISTINCT val), sum(DISTINCT id % 10)
FROM vector_aoco_test GROUP BY 1 ORDER BY 1;
SELECT id % 3 AS g, string_agg(val, ',' ORDER BY id)
FROM vector_aoco_small GROUP BY 1 ORDER BY 1;
SELECT id % 2 AS g, string_agg(val || ':' || id, '|' ORDER BY num DESC, id)
FROM vector_aoco_test WHERE id < 20 GROUP BY 1 ORDER BY 1;
SELECT id % 3 AS g, array_agg(id ORDER BY id) FILTER (WHERE val IS NOT NULL)
FROM vector_aoco_small GROUP BY 1 ORDER BY 1;
SELECT id % 3 AS g, count(DISTINCT id) FROM vector_aoco_small
GROUP BY 1 HAVING count(DISTINCT id) > 3 ORDER BY 1;
SELECT id % 100 AS g, array_agg(DISTINCT id % 10 ORDER BY id % 10)
FROM vector_aoco_spill_wide GROUP BY 1 ORDER BY 1 LIMIT 3;
EXPLAIN (COSTS OFF)
SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY id) FROM vector_aoco_small;
SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY id),
       percentile_disc(0.5) WITHIN GROUP (ORDER BY id)
FROM vector_aoco_small;
SELECT id % 3 AS g, percentile_cont(0.5) WITHIN GROUP (ORDER BY id)
FROM vector_aoco_small GROUP BY 1 ORDER BY 1;
SELECT percentile_cont(0.25) WITHIN GROUP (ORDER BY num),
       percentile_cont(0.75) WITHIN GROUP (ORDER BY num)
FROM vector_aoco_test;
SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY id)
FROM vector_aoco_small WHERE id < 0;
SELECT percentile_disc(0.5) WITHIN GROUP (ORDER BY val)
FROM vector_aoco_small WHERE id < 0;
SELECT percentile_cont(NULL) WITHIN GROUP (ORDER BY id) FROM vector_aoco_small;
SELECT l.id, l.l, r.id, r.r
FROM vector_join_l l JOIN vector_join_r r ON l.id = r.id
ORDER BY l.id, r.r;
SELECT l.id, l.l, r.id, r.r
FROM vector_join_l l LEFT JOIN vector_join_r r ON l.id = r.id
ORDER BY l.id, r.r;
SELECT l.id, l.l, r.id, r.r
FROM vector_join_l l RIGHT JOIN vector_join_r r ON l.id = r.id
ORDER BY l.id, r.r;
SELECT l.id, l.l, r.id, r.r
FROM vector_join_l l FULL JOIN vector_join_r r ON l.id = r.id
ORDER BY l.id, r.r;
SELECT l.id FROM vector_join_l l
WHERE EXISTS (SELECT 1 FROM vector_join_r r WHERE r.id = l.id)
ORDER BY l.id;
SELECT l.id FROM vector_join_l l
WHERE NOT EXISTS (SELECT 1 FROM vector_join_r r WHERE r.id = l.id)
ORDER BY l.id;
SET enable_hashjoin = off;
SET enable_nestloop = off;
SET enable_mergejoin = on;
SET optimizer_enable_hashjoin = off;
SET optimizer_enable_nljoin = off;
EXPLAIN (COSTS OFF)
SELECT l.id, l.l, r.id, r.r
FROM vector_join_l l JOIN vector_join_r r ON l.id = r.id
ORDER BY l.id, r.r;
SELECT l.id, l.l, r.id, r.r
FROM vector_join_l l JOIN vector_join_r r ON l.id = r.id
ORDER BY l.id, r.r;
SELECT l.id, l.l, r.id, r.r
FROM vector_join_l l LEFT JOIN vector_join_r r ON l.id = r.id
ORDER BY l.id, r.r;
SELECT l.id, l.l, r.id, r.r
FROM vector_join_l l RIGHT JOIN vector_join_r r ON l.id = r.id
ORDER BY l.id, r.r;
SELECT l.id, l.l, r.id, r.r
FROM vector_join_l l FULL JOIN vector_join_r r ON l.id = r.id
ORDER BY l.id, r.r;
SELECT l.id FROM vector_join_l l
WHERE EXISTS (SELECT 1 FROM vector_join_r r WHERE r.id = l.id)
ORDER BY l.id;
SELECT l.id FROM vector_join_l l
WHERE NOT EXISTS (SELECT 1 FROM vector_join_r r WHERE r.id = l.id)
ORDER BY l.id;
SELECT count(*) FROM vector_heap_test h JOIN vector_aoro_test r ON h.id = r.id;
RESET enable_hashjoin;
RESET enable_nestloop;
RESET enable_mergejoin;
RESET optimizer_enable_hashjoin;
RESET optimizer_enable_nljoin;
SET enable_hashjoin = off;
SET enable_mergejoin = off;
SET enable_nestloop = on;
SET optimizer_enable_hashjoin = off;
SET optimizer_enable_mergejoin = off;
SET optimizer_enable_nljoin = on;
EXPLAIN (COSTS OFF)
SELECT l.id, l.l, r.id, r.r
FROM vector_join_l l JOIN vector_join_r r ON l.id = r.id
ORDER BY l.id, r.r;
SELECT l.id, l.l, r.id, r.r
FROM vector_join_l l JOIN vector_join_r r ON l.id = r.id
ORDER BY l.id, r.r;
SELECT l.id, l.l, r.id, r.r
FROM vector_join_l l LEFT JOIN vector_join_r r ON l.id = r.id
ORDER BY l.id, r.r;
SELECT l.id FROM vector_join_l l
WHERE EXISTS (SELECT 1 FROM vector_join_r r WHERE r.id = l.id)
ORDER BY l.id;
SELECT l.id FROM vector_join_l l
WHERE NOT EXISTS (SELECT 1 FROM vector_join_r r WHERE r.id = l.id)
ORDER BY l.id;
SELECT l.id FROM vector_aoco_small l
WHERE l.id NOT IN (SELECT r.id FROM vector_join_r r) ORDER BY l.id;
SELECT a.id, a.val, b.id, b.val
FROM vector_aoco_small a CROSS JOIN vector_aoco_small b
WHERE a.id <= 2 AND b.id <= 3 ORDER BY a.id, b.id;
SELECT h.id, r.id FROM vector_heap_test h JOIN vector_aoro_test r ON h.id = r.id
ORDER BY h.id LIMIT 5;
RESET enable_hashjoin;
RESET enable_mergejoin;
RESET enable_nestloop;
RESET optimizer_enable_hashjoin;
RESET optimizer_enable_mergejoin;
RESET optimizer_enable_nljoin;
ALTER DATABASE regression SET statement_mem = '400kB';
SELECT count(t2.pad) FROM vector_aoco_wide t1 JOIN vector_aoco_wide t2 ON t1.id = t2.id;
SELECT count(t2.pad) FROM vector_aoco_wide t1 FULL JOIN vector_aoco_wide t2 ON t1.id = t2.id AND t2.id < 1000;
SET enable_hashjoin = off;
SET enable_nestloop = off;
SET enable_mergejoin = on;
SET optimizer_enable_hashjoin = off;
SET optimizer_enable_nljoin = off;
SELECT count(t2.pad) FROM vector_aoco_wide t1 JOIN vector_aoco_wide t2 ON t1.id = t2.id;
RESET enable_hashjoin;
RESET enable_nestloop;
RESET enable_mergejoin;
RESET optimizer_enable_hashjoin;
RESET optimizer_enable_nljoin;
SELECT grp, count(*), sum(length(pad)) FROM vector_aoco_spill GROUP BY grp ORDER BY grp;
SELECT count(*) FROM (SELECT id FROM vector_aoco_spill_wide GROUP BY id) s;
SELECT count(DISTINCT id), sum(DISTINCT id % 100) FROM vector_aoco_spill_wide;
SELECT id FROM vector_aoco_spill_wide ORDER BY pad, id LIMIT 3;
SELECT DISTINCT ON (grp) id FROM vector_unique_test ORDER BY grp, id LIMIT 3;
ALTER DATABASE regression RESET statement_mem;
EXPLAIN (COSTS OFF)
SELECT id, id * 2 + 1 AS doubled FROM vector_heap_test
WHERE id % 2000 = 0 ORDER BY id;
SELECT id, id * 2 + 1 AS doubled FROM vector_heap_test
WHERE id % 2000 = 0 ORDER BY id;
SELECT count(*), sum(big), min(num), max(num), count(val) FROM vector_heap_test;
EXPLAIN (COSTS OFF) SELECT count(*) FROM vector_aoro_test
WHERE val IS NULL OR num < 10;
SELECT count(*) FROM vector_aoro_test WHERE val IS NULL OR num < 10;
SELECT count(*), sum(h.id), sum(r.big)
FROM vector_heap_test h JOIN vector_aoro_test r ON h.id = r.id;
SELECT count(*) FROM vector_heap_test h
LEFT JOIN vector_aoro_test r ON h.id = r.id AND r.id < 100;
SET gp_vectorized_execution = auto;
SET gp_vectorized_min_rows = 1000000;
EXPLAIN (COSTS OFF) SELECT sum(id) FROM vector_aoco_small;
SET gp_vectorized_min_rows = 1;
EXPLAIN (COSTS OFF) SELECT sum(big), sum(id) FROM vector_aoco_test;
EXPLAIN (COSTS OFF) SELECT id FROM vector_aoco_test LIMIT 3;
EXPLAIN (COSTS OFF) SELECT id, id * 2 AS d FROM vector_aoco_test LIMIT 3;
SELECT sum(id * 2 + 1) FROM vector_aoco_test;
EXPLAIN (COSTS OFF) SELECT id, val FROM vector_aoco_test LIMIT 3;
SET statement_mem = '256kB';
EXPLAIN (COSTS OFF) SELECT val, count(*) FROM vector_aoco_test GROUP BY val;
RESET statement_mem;
RESET gp_vectorized_min_rows;
RESET gp_vectorized_execution;
DROP TABLE vector_aoco_test;
DROP TABLE vector_aoco_small;
DROP TABLE vector_join_l;
DROP TABLE vector_join_r;
DROP TABLE vector_aoco_wide;
DROP TABLE vector_aoco_spill;
DROP TABLE vector_aoco_spill_wide;
DROP TABLE vector_heap_test;
DROP TABLE vector_aoro_test;
DROP TABLE vector_limit_test;
DROP TABLE vector_unique_test;
