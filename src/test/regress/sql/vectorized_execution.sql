SET client_min_messages = warning;
DROP TABLE IF EXISTS vector_aoco_test;
DROP TABLE IF EXISTS vector_aoco_small;
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
RESET gp_vectorized_execution;
DROP TABLE vector_aoco_test;
DROP TABLE vector_aoco_small;
