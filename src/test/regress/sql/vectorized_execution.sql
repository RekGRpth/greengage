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
SELECT id, big + 1 AS b1 FROM vector_aoco_test ORDER BY id LIMIT 5;
SELECT id, CASE WHEN id % 2 = 0 THEN 'even' ELSE 'odd' END AS parity
FROM vector_aoco_test WHERE id BETWEEN 4998 AND 5010 ORDER BY id;
SELECT id, coalesce(val, 'NONE') FROM vector_aoco_test WHERE id % 500 = 0 ORDER BY id;
SELECT id FROM vector_aoco_test ORDER BY big DESC LIMIT 3;
SELECT id, (SELECT max(big) FROM vector_aoco_small s WHERE s.id = t.id) AS m
FROM vector_aoco_test t WHERE t.id < 12 ORDER BY id;
SELECT t.id, EXISTS (SELECT 1 FROM vector_aoco_small s WHERE s.id = t.id) AS has_s
FROM vector_aoco_test t WHERE t.id < 8 ORDER BY t.id;
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
ALTER DATABASE regression SET statement_mem = '400kB';
SELECT count(t2.pad) FROM vector_aoco_wide t1 JOIN vector_aoco_wide t2 ON t1.id = t2.id;
SELECT count(t2.pad) FROM vector_aoco_wide t1 FULL JOIN vector_aoco_wide t2 ON t1.id = t2.id AND t2.id < 1000;
SELECT grp, count(*), sum(length(pad)) FROM vector_aoco_spill GROUP BY grp ORDER BY grp;
SELECT count(*) FROM (SELECT id FROM vector_aoco_spill_wide GROUP BY id) s;
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
