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
DROP TABLE IF EXISTS vector_time_test;
DROP TABLE IF EXISTS vector_numeric_test;
DROP TABLE IF EXISTS vector_numeric_batch_test;
DROP TABLE IF EXISTS vector_numeric_wide_test;
DROP TABLE IF EXISTS vector_arithmetic_test;
DROP TABLE IF EXISTS vector_float_test;
DROP TABLE IF EXISTS vector_append_parent CASCADE;
DROP TABLE IF EXISTS vector_part_test CASCADE;
DROP TABLE IF EXISTS vector_drop_test CASCADE;
DROP TABLE IF EXISTS vector_union_a, vector_union_b, vector_union_h, vector_union_hj CASCADE;
DROP TABLE IF EXISTS vector_copy_test;
DROP TABLE IF EXISTS vector_copy_heap;
DROP TABLE IF EXISTS vector_copy_from;
DROP TABLE IF EXISTS vector_copy_from2;
DROP TABLE IF EXISTS vector_copy_sreh;
DROP TABLE IF EXISTS vector_copy_sreh2;
DROP TABLE IF EXISTS vector_copy_csv;
DROP TABLE IF EXISTS vector_nl_prefetch;
DROP TABLE IF EXISTS vector_boolnull;
DROP TABLE IF EXISTS vector_subplan_lifetime;
DROP TABLE IF EXISTS vector_nl_prefetch_agg;
DROP TABLE IF EXISTS vector_fixedref;
DROP TABLE IF EXISTS vector_nl_pull_a;
DROP TABLE IF EXISTS vector_nl_pull_b;
DROP TABLE IF EXISTS vector_cpf_1;
DROP TABLE IF EXISTS vector_cpf_2;
DROP TABLE IF EXISTS vector_cpf_3m;
DROP TABLE IF EXISTS vector_wholerow_t;
DROP VIEW IF EXISTS vector_wholerow_v;
DROP TYPE IF EXISTS vector_wholerow_rec;
DROP TYPE IF EXISTS vector_wholerow_nested;
DROP TABLE IF EXISTS vector_gsets_empty;
DROP TABLE IF EXISTS vector_gsets_rows;
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
CREATE TABLE vector_time_test (id int4, d date, ts timestamp, tz timestamptz)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (id);
INSERT INTO vector_time_test
SELECT g,
       CASE WHEN g % 7 = 0 THEN NULL ELSE DATE '2001-01-01' + g END,
       CASE WHEN g % 5 = 0 THEN NULL ELSE
            TIMESTAMP '2001-01-01 00:00:00' + g * INTERVAL '1 hour' END,
       CASE WHEN g % 3 = 0 THEN NULL ELSE
            TIMESTAMPTZ '2001-01-01 00:00:00+00' + g * INTERVAL '1 hour' END
FROM generate_series(1, 10) AS g;
ANALYZE vector_time_test;
CREATE TABLE vector_numeric_test (id int4, n numeric)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (id);
INSERT INTO vector_numeric_test
VALUES (1, -1.20), (2, 0), (3, 1.2), (4, 1.200), (5, 2.5),
       (6, 'NaN'), (7, 'NaN'), (8, NULL);
ANALYZE vector_numeric_test;
-- Exercises sum()/avg(numeric) over several batches: mixed folded and
-- display scales, mantissas beyond int64, NULLs, negatives, and grouping
-- with more distinct groups than one batch holds.
CREATE TABLE vector_numeric_batch_test (id int4, grp int4, n numeric)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (id);
INSERT INTO vector_numeric_batch_test
SELECT g, g % 5,
       CASE g % 8
         WHEN 0 THEN g::numeric
         WHEN 1 THEN round((g * 100 + 13) / 100, 2)
         WHEN 2 THEN g::numeric * 0.000000001
         WHEN 3 THEN 98765432109876543210987654321.123456789
         WHEN 4 THEN -round(g / 7.0, 4)
         WHEN 5 THEN NULL
         WHEN 6 THEN g::numeric * 1000000000000
         ELSE g::numeric / 7
       END
FROM generate_series(1, 7000) AS g;
ANALYZE vector_numeric_batch_test;
-- Wide numerics: more than 144 significant digits exceed the local digit
-- buffer of the inline (detoast-free) decode, and more than about 244
-- digits are stored with 4-byte varlena headers at all.
CREATE TABLE vector_numeric_wide_test (id int4, n numeric)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (id);
INSERT INTO vector_numeric_wide_test VALUES
(1, repeat('1234567890', 15)::numeric),
(2, repeat('9876543210', 30)::numeric),
(3, -repeat('1357913579', 15)::numeric / 3),
(4, NULL);
ANALYZE vector_numeric_wide_test;
CREATE TABLE vector_arithmetic_test (id int4, a int4, b int8, s int2)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (id);
INSERT INTO vector_arithmetic_test
VALUES (1, 2147483647, 9223372036854775807, 32767),
       (2, -2147483648, -9223372036854775808, -32768),
       (3, 12, 24, 12), (4, NULL, NULL, NULL);
ANALYZE vector_arithmetic_test;
CREATE TABLE vector_rle_test (id int4, grp int4, val text, num numeric(12,2), d date)
WITH (appendonly = true, orientation = column, compresstype = rle_type, compresslevel = 5)
DISTRIBUTED BY (id);
INSERT INTO vector_rle_test
SELECT g, (g / 500) * 500,
       CASE WHEN g % 97 = 0 THEN NULL ELSE 'run' || ((g / 500) * 500) END,
       CASE WHEN g % 89 = 0 THEN NULL ELSE ((g / 500) * 500)::numeric + 0.50 END,
       DATE '2001-01-01' + (g / 500)
FROM generate_series(1, 4000) AS g;
ANALYZE vector_rle_test;
CREATE TABLE vector_float_test (id int4, f4 float4, f8 float8)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (id);
INSERT INTO vector_float_test
VALUES (1, '-Infinity', '-Infinity'), (2, '-0.0', '-0.0'), (3, 0.0, 0.0),
       (4, 1.5, 1.5), (5, 'Infinity', 'Infinity'),
       (6, 'NaN', 'NaN'), (7, 'NaN', 'NaN'), (8, NULL, NULL);
ANALYZE vector_float_test;
-- Dense qual evaluation: packed INT4/DATE comparisons compiled into batch
-- result masks and a single selection compaction.  1003 rows cross the
-- 64-row mask words and leave a 3-row ragged tail; NULLs in q disqualify
-- the dense path batch-by-batch and fall back to the scalar filter.
CREATE TABLE vector_dense_test (id int4, d date, q int4)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (id);
INSERT INTO vector_dense_test
SELECT g, DATE '2020-01-01' + (g % 400),
       CASE WHEN g % 17 = 0 THEN NULL ELSE g % 250 END
FROM generate_series(1, 1003) AS g;
ANALYZE vector_dense_test;
SELECT min(q), max(q) FROM vector_dense_test WHERE q < 100;
SELECT min(q), max(q) FROM vector_dense_test WHERE q >= 10 AND q < 100;
SELECT min(d), max(d) FROM vector_dense_test
WHERE d >= DATE '2020-02-01' AND d < DATE '2020-06-01';
SELECT min(q), max(q) FROM vector_dense_test WHERE q = 3 OR q = 7;
SELECT q % 5 AS g, min(q), max(q) FROM vector_dense_test
WHERE q >= 20 AND q < 200 GROUP BY g ORDER BY g;
SELECT min(q), max(q) FROM vector_dense_test WHERE q IS NULL OR q >= 0;
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
CREATE TABLE vector_append_parent (id int4, big int8, val text, num numeric(12,2))
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (id);
CREATE TABLE vector_append_child1 () INHERITS (vector_append_parent);
CREATE TABLE vector_append_child2 () INHERITS (vector_append_parent)
WITH (appendonly = false);
CREATE TABLE vector_append_child3 () INHERITS (vector_append_parent);
INSERT INTO vector_append_child1
SELECT g, g::int8 * 10,
       CASE WHEN g % 7 = 0 THEN NULL ELSE 'v' || g END,
       (g % 100)::numeric / 3
FROM generate_series(1, 2500) AS g;
INSERT INTO vector_append_child2
SELECT g, g::int8 * 10,
       CASE WHEN g % 7 = 0 THEN NULL ELSE 'v' || g END,
       (g % 100)::numeric / 3
FROM generate_series(2501, 5000) AS g;
INSERT INTO vector_append_child3
SELECT g, g::int8 * 10,
       CASE WHEN g % 7 = 0 THEN NULL ELSE 'v' || g END,
       (g % 100)::numeric / 3
FROM generate_series(5001, 7500) AS g;
ANALYZE vector_append_child1;
ANALYZE vector_append_child2;
ANALYZE vector_append_child3;
-- Partitioned AOCO table: static and runtime pruning over vectorized
-- scans (the runtime shape scans surviving partitions as regions under
-- the pruning Append, including the empty-projection count form).
CREATE TABLE vector_part_test (id int4, grp int4)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (id) PARTITION BY RANGE (id);
CREATE TABLE vector_part_test_1 PARTITION OF vector_part_test
FOR VALUES FROM (1) TO (2001) WITH (appendonly = true, orientation = column);
CREATE TABLE vector_part_test_2 PARTITION OF vector_part_test
FOR VALUES FROM (2001) TO (4001) WITH (appendonly = true, orientation = column);
CREATE TABLE vector_part_test_3 PARTITION OF vector_part_test
FOR VALUES FROM (4001) TO (6001) WITH (appendonly = true, orientation = column);
INSERT INTO vector_part_test SELECT g, g % 10 FROM generate_series(1, 6000) AS g;
ANALYZE vector_part_test;
-- Schema evolution: dropping a column on the parent realigns every child
-- the same way, so the whole Append (parent scan included) still runs
-- inside one region with the aggregate.
CREATE TABLE vector_drop_test (id int4, grp int4, pad text, num numeric(12,2))
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (id);
CREATE TABLE vector_drop_child1 () INHERITS (vector_drop_test);
ALTER TABLE vector_drop_test DROP COLUMN pad;
INSERT INTO vector_drop_test
SELECT g, g % 10, (g % 50)::numeric / 3 FROM generate_series(1, 2000) AS g;
INSERT INTO vector_drop_child1
SELECT g, g % 10, (g % 50)::numeric / 3 FROM generate_series(2001, 4000) AS g;
ANALYZE vector_drop_test; ANALYZE vector_drop_child1;
-- UNION ALL over independently created relations whose column positions
-- differ: the divergent child fills a private batch and its columns move
-- onto the shared layout at each batch boundary.
CREATE TABLE vector_union_a (id int4, v numeric(12,2))
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (id);
CREATE TABLE vector_union_b (junk int8, id int4, v numeric(12,2))
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (id);
CREATE TABLE vector_union_h (id int4, v numeric(12,2)) DISTRIBUTED BY (id);
CREATE TABLE vector_union_hj (junk int8, id int4, v numeric(12,2)) DISTRIBUTED BY (id);
INSERT INTO vector_union_a SELECT g, (g % 50)::numeric / 3 FROM generate_series(1, 2000) AS g;
INSERT INTO vector_union_b SELECT g, g, (g % 50)::numeric / 3 FROM generate_series(2001, 4000) AS g;
INSERT INTO vector_union_h SELECT g, (g % 50)::numeric / 3 FROM generate_series(1, 2000) AS g;
INSERT INTO vector_union_hj SELECT g, g, (g % 50)::numeric / 3 FROM generate_series(2001, 4000) AS g;
ANALYZE vector_union_a; ANALYZE vector_union_b;
ANALYZE vector_union_h; ANALYZE vector_union_hj;
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
SELECT count(*) FILTER (WHERE id = 2500) AS eq,
       count(*) FILTER (WHERE id <> 2500) AS ne,
       count(*) FILTER (WHERE id < 2500) AS lt,
       count(*) FILTER (WHERE id <= 2500) AS le,
       count(*) FILTER (WHERE id > 2500) AS gt,
       count(*) FILTER (WHERE id >= 2500) AS ge
FROM vector_aoco_test;
SELECT count(*) FILTER (WHERE big = 25000) AS eq,
       count(*) FILTER (WHERE big <> 25000) AS ne,
       count(*) FILTER (WHERE big < 25000) AS lt,
       count(*) FILTER (WHERE big <= 25000) AS le,
       count(*) FILTER (WHERE big > 25000) AS gt,
       count(*) FILTER (WHERE big >= 25000) AS ge
FROM vector_aoco_test;
SELECT count(*) FILTER (WHERE (CASE WHEN id % 10 = 0 THEN NULL ELSE id END) = 2500)
       AS matching,
       count((CASE WHEN id % 10 = 0 THEN NULL ELSE id END) = 2500) AS nonnull
FROM vector_aoco_test;
SELECT count(*) FILTER (WHERE d = DATE '2001-01-06') AS eq,
       count(*) FILTER (WHERE d <> DATE '2001-01-06') AS ne,
       count(*) FILTER (WHERE d < DATE '2001-01-06') AS lt,
       count(*) FILTER (WHERE d <= DATE '2001-01-06') AS le,
       count(*) FILTER (WHERE d > DATE '2001-01-06') AS gt,
       count(*) FILTER (WHERE d >= DATE '2001-01-06') AS ge
FROM vector_time_test;
SELECT count(*) FILTER (WHERE ts = TIMESTAMP '2001-01-01 05:00:00') AS eq,
       count(*) FILTER (WHERE ts <> TIMESTAMP '2001-01-01 05:00:00') AS ne,
       count(*) FILTER (WHERE ts < TIMESTAMP '2001-01-01 05:00:00') AS lt,
       count(*) FILTER (WHERE ts <= TIMESTAMP '2001-01-01 05:00:00') AS le,
       count(*) FILTER (WHERE ts > TIMESTAMP '2001-01-01 05:00:00') AS gt,
       count(*) FILTER (WHERE ts >= TIMESTAMP '2001-01-01 05:00:00') AS ge
FROM vector_time_test;
SELECT count(*) FILTER (WHERE tz = TIMESTAMPTZ '2001-01-01 05:00:00+00') AS eq,
       count(*) FILTER (WHERE tz <> TIMESTAMPTZ '2001-01-01 05:00:00+00') AS ne,
       count(*) FILTER (WHERE tz < TIMESTAMPTZ '2001-01-01 05:00:00+00') AS lt,
       count(*) FILTER (WHERE tz <= TIMESTAMPTZ '2001-01-01 05:00:00+00') AS le,
       count(*) FILTER (WHERE tz > TIMESTAMPTZ '2001-01-01 05:00:00+00') AS gt,
       count(*) FILTER (WHERE tz >= TIMESTAMPTZ '2001-01-01 05:00:00+00') AS ge
FROM vector_time_test;
SELECT count(*) FILTER (WHERE n = 1.20) AS eq,
       count(*) FILTER (WHERE n <> 1.20) AS ne,
       count(*) FILTER (WHERE n < 1.20) AS lt,
       count(*) FILTER (WHERE n <= 1.20) AS le,
       count(*) FILTER (WHERE n > 1.20) AS gt,
       count(*) FILTER (WHERE n >= 1.20) AS ge
FROM vector_numeric_test;
SELECT count(*) FILTER (WHERE n = 'NaN') AS eq,
       count(*) FILTER (WHERE n <> 'NaN') AS ne,
       count(*) FILTER (WHERE n < 'NaN') AS lt,
       count(*) FILTER (WHERE n <= 'NaN') AS le,
       count(*) FILTER (WHERE n > 'NaN') AS gt,
       count(*) FILTER (WHERE n >= 'NaN') AS ge
FROM vector_numeric_test;
SELECT a + 5, a - 5, a * 5, a / 5, a % 5,
       b + 5, b - 5, b * 5, b / 5, b % 5
FROM vector_arithmetic_test WHERE id = 3;
SELECT sum(s), min(s), max(s) FROM vector_arithmetic_test;
SELECT sum(b), min(b), max(b) FROM vector_arithmetic_test;
SELECT avg(s), avg(a), avg(b) FROM vector_arithmetic_test;
-- Separate packed INT4 pipeline: NULL handling, builtins and arithmetic errors.
SELECT sum(a), avg(a), min(a), max(a) FROM vector_arithmetic_test;
SELECT sum(a + 1) FROM vector_arithmetic_test;
SELECT sum(a * 2) FROM vector_arithmetic_test;
SELECT sum(a), avg(a) FROM vector_arithmetic_test WHERE a > 0 AND a < 100;
SELECT a, sum(a), avg(a), min(a), max(a) FROM vector_arithmetic_test
GROUP BY a ORDER BY a NULLS LAST;
-- Packed INT2: no arithmetic, only comparisons, builtins and grouping.
SELECT sum(s), avg(s), min(s), max(s) FROM vector_arithmetic_test;
SELECT sum(s), avg(s) FROM vector_arithmetic_test WHERE s > 0 AND s < 100;
SELECT s, sum(s), avg(s), min(s), max(s) FROM vector_arithmetic_test
GROUP BY s ORDER BY s NULLS LAST;
-- INT24 arithmetic: int2 left, int4 right/result and overflow behavior.
SELECT sum(s * 2), min(s * 2), max(s * 2) FROM vector_arithmetic_test;
SELECT sum(s + 2147480000) FROM vector_arithmetic_test;
SELECT sum(s * 100000) FROM vector_arithmetic_test;
SELECT sum(s / 0) FROM vector_arithmetic_test;
-- INT42 and same-type INT2 arithmetic.
SELECT sum(2 * s), min(2 * s), max(2 * s) FROM vector_arithmetic_test;
SELECT sum(s * 1::int2), min(s * 1::int2), max(s * 1::int2) FROM vector_arithmetic_test;
SELECT sum(2147480000 + s) FROM vector_arithmetic_test;
SELECT sum(s * 100::int2) FROM vector_arithmetic_test;
SELECT sum(s % 0::int2) FROM vector_arithmetic_test;
-- Packed native Sort: DESC and NULLS LAST for every packed key type.
SELECT a FROM vector_arithmetic_test ORDER BY a DESC NULLS LAST;
SELECT s FROM vector_arithmetic_test ORDER BY s DESC NULLS LAST;
SELECT f4 FROM vector_float_test ORDER BY f4 DESC NULLS LAST;
SELECT d FROM vector_time_test ORDER BY d DESC NULLS LAST;
-- A custom btree opclass on a packed type keeps its own ordering: with a
-- non-default ordering operator the column leaves the packed layout, and
-- the SortSupport comparator decides.
CREATE FUNCTION vector_int4_rev_cmp(int4, int4) RETURNS int4
LANGUAGE sql IMMUTABLE AS
$$ SELECT CASE WHEN $1 = $2 THEN 0 WHEN $1 > $2 THEN -1 ELSE 1 END $$;
CREATE OPERATOR <<< (LEFTARG = int4, RIGHTARG = int4, PROCEDURE = int4gt);
CREATE OPERATOR <<= (LEFTARG = int4, RIGHTARG = int4, PROCEDURE = int4ge);
CREATE OPERATOR >>= (LEFTARG = int4, RIGHTARG = int4, PROCEDURE = int4le);
CREATE OPERATOR >>> (LEFTARG = int4, RIGHTARG = int4, PROCEDURE = int4lt);
CREATE OPERATOR CLASS vector_int4_rev_ops FOR TYPE int4 USING btree AS
	OPERATOR 1 <<<,
	OPERATOR 2 <<=,
	OPERATOR 3 =,
	OPERATOR 4 >>=,
	OPERATOR 5 >>>,
	FUNCTION 1 vector_int4_rev_cmp(int4, int4);
CREATE TABLE vector_rev_sort (a int4, k int4)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (k);
INSERT INTO vector_rev_sort SELECT g, 1 FROM generate_series(1, 50) g;
SET gp_vectorized_execution = force;
SELECT a FROM vector_rev_sort ORDER BY a USING <<< NULLS LAST LIMIT 5;
SET gp_vectorized_execution = force;
DROP TABLE vector_rev_sort;
DROP OPERATOR CLASS vector_int4_rev_ops USING btree;
DROP OPERATOR <<< (int4, int4);
DROP OPERATOR <<= (int4, int4);
DROP OPERATOR >>= (int4, int4);
DROP OPERATOR >>> (int4, int4);
DROP FUNCTION vector_int4_rev_cmp(int4, int4);
SELECT avg(b) FROM vector_arithmetic_test WHERE id = 4;
SELECT var_pop(id::int2), var_samp(id::int2), stddev(id::int2),
       stddev_pop(id::int2), stddev_samp(id::int2)
FROM vector_aoco_small;
SELECT var_pop(id), var_samp(id), stddev(id), stddev_pop(id), stddev_samp(id)
FROM vector_aoco_small;
SELECT var_pop(id::int8), var_samp(id::int8), stddev(id::int8),
       stddev_pop(id::int8), stddev_samp(id::int8)
FROM vector_aoco_small;
SELECT avg(num) FROM vector_aoco_test;
SELECT avg(n), sum(n) FROM vector_numeric_test;
SELECT sum(n * 2), avg(n - 1) FROM vector_numeric_test;
SELECT sum(n), avg(n), min(n), max(n), count(n) FROM vector_numeric_batch_test;
-- Numeric arithmetic arguments take the deferred (decoded, no Numeric
-- materialization) path into sum/avg, with per-row fallback mixed in.
SELECT sum(n * 2), avg(n + 1), sum(n - 0.5), sum(n * n)
FROM vector_numeric_batch_test;
-- Nested arithmetic: the inner nodes stay decoded for the parent too.
SELECT id, n * (1 - n) FROM vector_numeric_test ORDER BY id;
SELECT sum(n * (1 - n)), avg((1 - n) * (1 + n)) FROM vector_numeric_batch_test;
SELECT grp, sum(n), avg(n), count(n)
FROM vector_numeric_batch_test GROUP BY grp ORDER BY grp;
-- Interleaved keys and NULL keys exercise the per-batch group memo on its
-- miss path (every other row changes the group, NULL groups with NULL).
SELECT id % 3 AS k, count(*), sum(n) FROM vector_numeric_batch_test GROUP BY k ORDER BY k;
SELECT grp, id % 2 AS parity, count(*) FROM vector_numeric_batch_test GROUP BY grp, parity ORDER BY grp, parity;
SELECT count(*) FROM (SELECT CASE WHEN id % 5 = 0 THEN NULL ELSE id % 5 END AS k, sum(id) FROM vector_numeric_batch_test GROUP BY k) t;
SELECT count(*), sum(s) FROM
(SELECT id, sum(n) AS s FROM vector_numeric_batch_test GROUP BY id) t;
SELECT sum(n) FILTER (WHERE id % 2 = 0),
       avg(n) FILTER (WHERE id % 3 = 0)
FROM vector_numeric_batch_test;
SELECT sum(n), avg(n), min(n), max(n), count(n)
FROM vector_numeric_wide_test;
SELECT id, n > repeat('1', 150)::numeric, n < 0
FROM vector_numeric_wide_test ORDER BY id;
SELECT min(n), max(n) FROM vector_numeric_test;
SELECT n, n + 1, n - 1, n * 3, n + n, n * n FROM vector_numeric_test;
SELECT sum(b) FROM vector_arithmetic_test WHERE id = 4;
SELECT sum(f4), sum(f8) FROM vector_float_test WHERE id = 2;
SELECT id, f8 + f8, f8 - f8, f8 * f8, f8 / f8
FROM vector_float_test WHERE id IN (1, 2, 4, 5, 8) ORDER BY id;
SELECT id, f4 + f4, f4 - f4, f4 * f4, f4 / f4
FROM vector_float_test WHERE id IN (1, 2, 4, 5, 8) ORDER BY id;
SELECT sum(f8 * 2 - f8), avg(f4 + f4)
FROM vector_float_test WHERE id BETWEEN 2 AND 5;
-- Packed FLOAT4: arithmetic, NaN/Inf/-0 comparisons and hashed grouping.
SELECT sum(f4 * 2), min(f4), max(f4) FROM vector_float_test;
SELECT sum(f4 * 2), min(f4), max(f4) FROM vector_float_test
WHERE f4 > 0.0 AND f4 < 'Infinity';
SELECT f4, min(f4), max(f4) FROM vector_float_test
GROUP BY f4 ORDER BY f4 NULLS LAST;
-- sum(float4) has transtype float4 and no finalfn: every accumulation
-- step must round to float4 the way float4pl does, not accumulate in
-- float8 and truncate at the end.
CREATE TABLE vector_f4sum (x float4, k int4)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (k);
INSERT INTO vector_f4sum VALUES (16777216.0, 1);
INSERT INTO vector_f4sum VALUES (1.0, 1);
INSERT INTO vector_f4sum VALUES (1.0, 1);
INSERT INTO vector_f4sum VALUES (2.5, 1);
INSERT INTO vector_f4sum VALUES (-0.5, 1);
SET gp_vectorized_execution = force;
SELECT sum(x) FROM vector_f4sum;
SELECT sum(x * 2) FROM vector_f4sum;
SELECT sum(x) FROM vector_f4sum WHERE x > 100;
SET gp_vectorized_execution = off;
SELECT sum(x) FROM vector_f4sum;
SELECT sum(x * 2) FROM vector_f4sum;
SELECT sum(x) FROM vector_f4sum WHERE x > 100;
SET gp_vectorized_execution = force;
-- ---- vectorized aggregates under JIT: a FILTER rejecting every row of a
-- batch must not reach the compiled evaluation loop with a zero selection ----
SET jit = on;
SET jit_above_cost = 0;
SELECT sum(x * 2) FILTER (WHERE false) FROM vector_f4sum;
SELECT sum(x * 2) FILTER (WHERE x > 100) FROM vector_f4sum;
RESET jit;
RESET jit_above_cost;
SELECT f8 / 0 FROM vector_float_test WHERE id = 4;
SELECT f4 / 0 FROM vector_float_test WHERE id = 4;
SELECT min(f4), max(f4), min(f8), max(f8) FROM vector_float_test;
SELECT min(d), max(d), min(ts), max(ts), min(tz), max(tz) FROM vector_time_test;
-- Packed 32-bit DATE: range comparison, min/max and a hashed NULL key.
SELECT min(d), max(d) FROM vector_time_test
WHERE d >= DATE '2001-01-03' AND d < DATE '2001-01-09';
SELECT d, min(d), max(d) FROM vector_time_test
GROUP BY d ORDER BY d NULLS LAST;
SELECT array_agg(CASE WHEN id = 2 THEN NULL ELSE id END ORDER BY id)
FROM vector_aoco_small WHERE id <= 3;
SELECT count(a + 1), count(b + 1) FROM vector_arithmetic_test;
SELECT a + 1 FROM vector_arithmetic_test WHERE id = 1;
SELECT a - 1 FROM vector_arithmetic_test WHERE id = 2;
SELECT a * 2 FROM vector_arithmetic_test WHERE id = 1;
SELECT a / -1 FROM vector_arithmetic_test WHERE id = 2;
SELECT a % -1 FROM vector_arithmetic_test WHERE id = 2;
SELECT b + 1 FROM vector_arithmetic_test WHERE id = 1;
SELECT b - 1 FROM vector_arithmetic_test WHERE id = 2;
SELECT b * 2 FROM vector_arithmetic_test WHERE id = 1;
SELECT b / -1 FROM vector_arithmetic_test WHERE id = 2;
SELECT b % -1 FROM vector_arithmetic_test WHERE id = 2;
SELECT a / 0 FROM vector_arithmetic_test WHERE id = 3;
SELECT a % 0 FROM vector_arithmetic_test WHERE id = 3;
SELECT count(*), count(val), count(num), sum(num), min(d), max(d), min(grp), max(grp)
FROM vector_rle_test;
SELECT grp, count(*), sum(num), min(val), max(d) FROM vector_rle_test GROUP BY grp ORDER BY grp;
SELECT count(*), sum(num) FROM vector_rle_test WHERE d >= DATE '2001-01-05' AND num > 100;
SELECT id, val, num, d FROM vector_rle_test WHERE id % 997 = 0 ORDER BY id;
SELECT b / 0 FROM vector_arithmetic_test WHERE id = 3;
SELECT b % 0 FROM vector_arithmetic_test WHERE id = 3;
SELECT count(*) FILTER (WHERE f4 = 1.5) AS eq,
       count(*) FILTER (WHERE f4 <> 1.5) AS ne,
       count(*) FILTER (WHERE f4 < 1.5) AS lt,
       count(*) FILTER (WHERE f4 <= 1.5) AS le,
       count(*) FILTER (WHERE f4 > 1.5) AS gt,
       count(*) FILTER (WHERE f4 >= 1.5) AS ge
FROM vector_float_test;
SELECT count(*) FILTER (WHERE f4 = 'NaN') AS eq,
       count(*) FILTER (WHERE f4 <> 'NaN') AS ne,
       count(*) FILTER (WHERE f4 < 'NaN') AS lt,
       count(*) FILTER (WHERE f4 <= 'NaN') AS le,
       count(*) FILTER (WHERE f4 > 'NaN') AS gt,
       count(*) FILTER (WHERE f4 >= 'NaN') AS ge
FROM vector_float_test;
SELECT count(*) FILTER (WHERE f8 = 1.5) AS eq,
       count(*) FILTER (WHERE f8 <> 1.5) AS ne,
       count(*) FILTER (WHERE f8 < 1.5) AS lt,
       count(*) FILTER (WHERE f8 <= 1.5) AS le,
       count(*) FILTER (WHERE f8 > 1.5) AS gt,
       count(*) FILTER (WHERE f8 >= 1.5) AS ge
FROM vector_float_test;
SELECT count(*) FILTER (WHERE f8 = 'NaN') AS eq,
       count(*) FILTER (WHERE f8 <> 'NaN') AS ne,
       count(*) FILTER (WHERE f8 < 'NaN') AS lt,
       count(*) FILTER (WHERE f8 <= 'NaN') AS le,
       count(*) FILTER (WHERE f8 > 'NaN') AS gt,
       count(*) FILTER (WHERE f8 >= 'NaN') AS ge
FROM vector_float_test;
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
SELECT t.id, NOT EXISTS (SELECT 1 FROM vector_aoco_small s WHERE s.id = t.id) AS no_s
FROM vector_aoco_test t WHERE t.id < 8 ORDER BY t.id;
SELECT t.id, (SELECT count(*) FROM vector_aoco_small s WHERE s.id = t.id) AS cnt
FROM vector_aoco_test t WHERE t.id < 8 ORDER BY t.id;
-- Correlated subplans in scan expressions: re-executed per batch row via
-- ExecSubPlan, with the current row supplied as both scan and outer slot.
SELECT count(*) FROM vector_aoco_test t
WHERE EXISTS (SELECT 1 FROM vector_aoco_small s WHERE s.id = t.id);
SELECT count(*) FROM vector_aoco_test t
WHERE NOT EXISTS (SELECT 1 FROM vector_aoco_small s WHERE s.id = t.id);
SELECT count(*) FROM vector_aoco_test t
WHERE t.id IN (SELECT s.id FROM vector_aoco_small s WHERE s.id < t.id + 3);
SELECT sum(num) FROM vector_aoco_test t
WHERE EXISTS (SELECT 1 FROM vector_aoco_small s WHERE s.id = t.id);
-- Correlated subplans in aggregate arguments and FILTER use the aggregate's
-- outer targetlist layout when supplying the current row to ExecSubPlan.
SET gp_eager_two_phase_agg = on;
SELECT id % 3 AS g,
       sum((SELECT count(*) FROM vector_aoco_small s WHERE s.id = t.id)) AS matches,
       avg((SELECT max(s.id) FROM vector_aoco_small s WHERE s.id = t.id % 12)) AS avg_id
FROM vector_aoco_test t WHERE t.id < 100 GROUP BY g ORDER BY g;
SELECT id % 3 AS g, sum(num) FILTER (WHERE EXISTS
       (SELECT 1 FROM vector_aoco_small s WHERE s.id = t.id)) AS sum_num
FROM vector_aoco_test t WHERE t.id < 100 GROUP BY g ORDER BY g;
SET enable_hashagg = off;
SELECT id % 3 AS g, count(DISTINCT
       (SELECT s.id % 3 FROM vector_aoco_small s WHERE s.id = t.id % 12)) AS nd
FROM vector_aoco_test t WHERE t.id < 100 GROUP BY g ORDER BY g;
RESET enable_hashagg;
RESET gp_eager_two_phase_agg;
EXPLAIN (COSTS OFF)
SELECT id FROM vector_aoco_small WHERE id > (SELECT max(id) FROM vector_aoco_small) - 2 ORDER BY id;
SELECT id FROM vector_aoco_small WHERE id > (SELECT max(id) FROM vector_aoco_small) - 2 ORDER BY id;
SELECT (SELECT max(id) FROM vector_aoco_small) AS m;
SELECT count(*) FROM vector_aoco_test WHERE big > (SELECT max(id) FROM vector_aoco_small);
SELECT count(*) FROM vector_aoco_test WHERE id IN (SELECT id FROM vector_aoco_small);
SELECT count(*) FROM vector_aoco_small WHERE id NOT IN (SELECT id FROM vector_join_r);
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
SET enable_hashagg = off;
SELECT id % 2 AS a, id % 3 AS b, GROUPING(id % 2, id % 3) AS g,
       count(*), sum(id), min(id), max(id)
FROM vector_aoco_small GROUP BY ROLLUP(a, b) ORDER BY a, b, g;
SELECT id % 2 AS a, id % 3 AS b, GROUPING(id % 2, id % 3) AS g, count(*)
FROM vector_aoco_small GROUP BY CUBE(a, b) ORDER BY a, b, g;
SELECT id % 2 AS a, id % 3 AS b, GROUPING(id % 2, id % 3) AS g, count(*)
FROM vector_aoco_small GROUP BY GROUPING SETS ((a, b), (a), ()) ORDER BY a, b, g;
SELECT id % 2 AS a, count(*) FROM vector_aoco_small GROUP BY ROLLUP(a) ORDER BY a;
SELECT id % 2 AS a, id % 3 AS b, GROUPING(id % 2, id % 3) AS g, count(*)
FROM vector_aoco_small WHERE id < 0 GROUP BY ROLLUP(a, b) ORDER BY a, b, g;
SELECT id % 3 AS a, id % 4 AS b, GROUPING(id % 3, id % 4) AS g,
       count(*), sum(id)
FROM vector_aoco_test GROUP BY ROLLUP(a, b) HAVING count(*) > 1000
ORDER BY a, b, g;
SELECT id % 2 AS a, GROUPING(id % 2) AS g,
       string_agg(val, ',' ORDER BY id) AS vals
FROM vector_aoco_small GROUP BY ROLLUP(a) ORDER BY a, g;
RESET enable_hashagg;
-- Multi-phase sorted aggregation: each later phase replays the input from
-- its own tuplesort.  The 7000-row input crosses spool batch boundaries.
SET enable_hashagg = off;
SET optimizer_enable_hashagg = off;
SET gp_eager_two_phase_agg = on;
SET optimizer_force_multistage_agg = on;
EXPLAIN (COSTS OFF)
SELECT id % 20 AS a, id % 10 AS b, GROUPING(id % 20, id % 10) AS g,
       count(*), sum(id)
FROM vector_numeric_batch_test GROUP BY GROUPING SETS ((a, b), (a), (b), ());
SELECT id % 20 AS a, id % 10 AS b, GROUPING(id % 20, id % 10) AS g,
       count(*), sum(id)
FROM vector_numeric_batch_test GROUP BY GROUPING SETS ((a, b), (a), (b), ())
ORDER BY a NULLS FIRST, b NULLS FIRST, g;
SELECT g, count(*) AS ngroups, sum(cnt) AS input_rows, sum(sum_id) AS sum_id
FROM (SELECT id % 20 AS a, id % 10 AS b, id % 3 AS c,
             GROUPING(id % 20, id % 10, id % 3) AS g,
             count(*) AS cnt, sum(id) AS sum_id
      FROM vector_numeric_batch_test
      GROUP BY GROUPING SETS ((a,b,c),(a,b),(a,c),(b,c),(a),(b),(c),())) s
GROUP BY g ORDER BY g;
SELECT count(*) AS ngroups, sum(cnt) AS input_rows
FROM (SELECT id % 20 AS a, id % 10 AS b, id % 3 AS c, count(*) AS cnt
      FROM vector_numeric_batch_test
      GROUP BY GROUPING SETS ((a,b,c),(a,b),(a,c),(b,c),(a),(b),(c),())
      HAVING count(*) > 25) s;
SELECT id % 2 AS a, id % 3 AS b, GROUPING(id % 2, id % 3) AS g,
       count(*), sum(id)
FROM vector_aoco_small WHERE id < 0 GROUP BY GROUPING SETS ((a, b), (a), (b), ())
ORDER BY a, b, g;
SELECT id % 3 AS a, id % 2 AS b, GROUPING(id % 3, id % 2) AS g,
       count(*), sum((SELECT count(*) FROM vector_aoco_small s WHERE s.id = t.id)) AS matches
FROM vector_aoco_test t WHERE t.id < 100 GROUP BY GROUPING SETS ((a, b), (a), (b), ())
ORDER BY a NULLS FIRST, b NULLS FIRST, g;
RESET gp_eager_two_phase_agg;
RESET optimizer_force_multistage_agg;
RESET optimizer_enable_hashagg;
-- DISTINCT and ORDER BY aggregates have a separate sort state per grouping set.
SELECT id % 3 AS a, id % 2 AS b, GROUPING(id % 3, id % 2) AS g,
       count(DISTINCT id % 3) AS dc, string_agg(val, ',' ORDER BY id) AS vals
FROM vector_aoco_small GROUP BY GROUPING SETS ((a, b), (a), ())
ORDER BY a NULLS FIRST, b NULLS FIRST, g;
RESET enable_hashagg;
-- A sorted aggregate over a join chain: the region drives the Sort between
-- the join output batches and the sorted aggregation input.  The 4000-row
-- join output crosses the batch boundary inside a group.
SET enable_hashagg = off;
SET gp_eager_two_phase_agg = on;
EXPLAIN (COSTS OFF)
SELECT s.id % 3 AS g, count(*), sum(l.id), min(l.num)
FROM vector_aoco_test l JOIN vector_aoco_small s ON l.id % 10 = s.id - 1
GROUP BY g ORDER BY g;
SELECT s.id % 3 AS g, count(*), sum(l.id), min(l.num)
FROM vector_aoco_test l JOIN vector_aoco_small s ON l.id % 10 = s.id - 1
GROUP BY g ORDER BY g;
SELECT s.id % 2 AS g, count(*), sum(l.num)
FROM vector_aoco_test l LEFT JOIN vector_aoco_small s ON l.id % 10 = s.id - 1
GROUP BY g ORDER BY g;
RESET gp_eager_two_phase_agg;
RESET enable_hashagg;
-- A hash join whose probe side is a Redistribute Motion: the region feeds
-- its probe batches from the received producer rows, and both scans keep
-- the computed join keys in their targetlists.
EXPLAIN (COSTS OFF)
SELECT count(*) FROM vector_aoco_test l JOIN vector_aoco_wide w
  ON l.id % 1000 = w.id % 1000;
SELECT count(*) FROM vector_aoco_test l JOIN vector_aoco_wide w
  ON l.id % 1000 = w.id % 1000;
SELECT count(*) FROM vector_aoco_test l LEFT JOIN vector_aoco_wide w
  ON l.id % 1000 = w.id % 1000;
-- RIGHT join over a join chain (the aggregate sits between the join and
-- the region root): probe rows with no build match are dropped, and the
-- finish phase emits unmatched build rows null-extended across output
-- batch boundaries.
EXPLAIN (COSTS OFF)
SELECT cnt, count(*) AS custdist FROM
  (SELECT l.id AS lk, count(w.id) AS cnt
   FROM vector_aoco_wide w RIGHT JOIN vector_aoco_test l
     ON w.id % 2000 = l.id
   GROUP BY l.id) s
GROUP BY cnt ORDER BY custdist DESC, cnt DESC;
SELECT cnt, count(*) AS custdist FROM
  (SELECT l.id AS lk, count(w.id) AS cnt
   FROM vector_aoco_wide w RIGHT JOIN vector_aoco_test l
     ON w.id % 2000 = l.id
   GROUP BY l.id) s
GROUP BY cnt ORDER BY custdist DESC, cnt DESC;
-- Mixed aggregation (hashed grouping sets plus the plain empty set):
-- every row advances every set in one pass, plain sets project first.
SELECT id % 3 AS a, id % 2 AS b, GROUPING(id % 3, id % 2) AS g,
       count(*), sum(big), min(big), max(big)
FROM vector_aoco_small GROUP BY ROLLUP (a, b) ORDER BY a, b, g;
SELECT id % 3 AS a, id % 2 AS b, GROUPING(id % 3, id % 2) AS g, count(*)
FROM vector_aoco_small GROUP BY CUBE (a, b) ORDER BY a, b, g;
SELECT id % 3 AS a, id % 2 AS b, GROUPING(id % 3, id % 2) AS g, count(*)
FROM vector_aoco_small GROUP BY GROUPING SETS ((a, b), (a), ()) ORDER BY a, b, g;
SELECT count(*), sum(big) FROM vector_aoco_small GROUP BY GROUPING SETS (());
SET statement_mem = '512kB';
SELECT id % 1000 AS a, count(*), sum(id)
FROM vector_aoco_test GROUP BY ROLLUP (a) ORDER BY a;
RESET statement_mem;
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
EXPLAIN (COSTS OFF) SELECT count(*) FROM vector_append_parent;
SELECT count(*), sum(id), min(id), max(id) FROM vector_append_parent;
SELECT id % 3 AS g, count(*), sum(num)
FROM vector_append_parent WHERE id % 5 <> 0 GROUP BY g ORDER BY g;
EXPLAIN (COSTS OFF) SELECT id, big FROM vector_append_parent WHERE id < 10 ORDER BY id;
SELECT id, big FROM vector_append_parent WHERE id < 10 ORDER BY id;
-- Static pruning keeps the surviving partition's scan inside a region.
SELECT count(*), sum(grp) FROM vector_part_test WHERE id BETWEEN 2100 AND 3900;
-- Runtime pruning (param bounds): surviving partitions scan as regions
-- under the pruning Append, with and without a projection.
SELECT count(*) FROM vector_part_test
WHERE id BETWEEN (SELECT min(id) FROM vector_part_test WHERE grp = 3)
            AND (SELECT max(id) FROM vector_part_test WHERE grp = 3);
SELECT count(*), sum(grp) FROM vector_part_test
WHERE id BETWEEN (SELECT min(id) FROM vector_part_test WHERE grp = 3)
            AND (SELECT max(id) FROM vector_part_test WHERE grp = 3);
-- No pruning at all: the whole Append runs inside one region.
SELECT count(*) FROM vector_part_test WHERE grp = 7;
-- Pruning inside one region: an empty valid set yields no rows, and exec
-- pruning is recomputed for each nested-loop outer row.
SELECT count(*) FROM vector_part_test WHERE id > 10000;
SET enable_hashjoin = off;
SET enable_mergejoin = off;
SELECT count(*) FROM (VALUES (1), (2), (3)) v(k)
JOIN vector_part_test p ON p.id BETWEEN v.k * 1000 AND v.k * 1000 + 50;
RESET enable_hashjoin;
RESET enable_mergejoin;
SELECT count(*), sum(num) FROM vector_drop_test WHERE id % 3 = 0;
SELECT grp, count(*), sum(num) FROM vector_drop_test GROUP BY grp ORDER BY grp;
SELECT count(*), sum(v) FROM
(SELECT id, v FROM vector_union_a UNION ALL SELECT id, v FROM vector_union_b) t;
SELECT count(*), sum(v) FROM
(SELECT id, v FROM vector_union_h WHERE id % 3 = 0
 UNION ALL SELECT id, v FROM vector_union_hj WHERE id % 3 = 0) t;
-- Correlated subplans in Append children: identical inheritance layouts run
-- in one region; a divergent child uses its own tupdesc plus the attno remap.
SELECT count(*) FROM vector_append_parent p
WHERE p.id < 100 AND (SELECT count(*) FROM vector_aoco_small s WHERE s.id = p.id) > 0;
SELECT count(*) FROM vector_append_parent p
WHERE p.id < 100 AND EXISTS (SELECT 1 FROM vector_aoco_small s WHERE s.id = p.id);
SELECT id FROM
(SELECT id, v FROM vector_union_a UNION ALL SELECT id, v FROM vector_union_b) t
WHERE t.id < 100 AND (SELECT count(*) FROM vector_aoco_small s WHERE s.id = t.id) > 0
ORDER BY id;
-- With an empty targetlist the divergent mapping is unavailable, so the
-- Append stays on the row executor above independent per-child regions.
EXPLAIN (COSTS OFF)
SELECT count(*) FROM
(SELECT id, v FROM vector_union_a UNION ALL SELECT id, v FROM vector_union_b) t
WHERE t.id < 100 AND (SELECT count(*) FROM vector_aoco_small s WHERE s.id = t.id) > 0;
SELECT count(*) FROM
(SELECT id, v FROM vector_union_a UNION ALL SELECT id, v FROM vector_union_b) t
WHERE t.id < 100 AND (SELECT count(*) FROM vector_aoco_small s WHERE s.id = t.id) > 0;
-- EXPLAIN resolves motion keys through custom_ps and the Agg display chain.
EXPLAIN (COSTS OFF)
SELECT l.id, sum(r.id) FROM vector_join_l l JOIN vector_join_r r ON l.id = r.id
GROUP BY l.id ORDER BY l.id;
SELECT l.id, sum(r.id) FROM vector_join_l l JOIN vector_join_r r ON l.id = r.id
GROUP BY l.id ORDER BY l.id;
-- By-ref values survive HashJoin until the aggregate expression consumes them.
SELECT sum(CASE WHEN r.r LIKE 'r3%' THEN length(l.l) ELSE 0 END)
FROM vector_join_l l JOIN vector_join_r r ON l.id = r.id;
SET gp_vectorized_execution = auto;
SET gp_vectorized_min_rows = 1000000;
EXPLAIN (COSTS OFF) SELECT sum(id) FROM vector_aoco_small;
SET gp_vectorized_min_rows = 1;
EXPLAIN (COSTS OFF) SELECT sum(big), sum(id) FROM vector_aoco_test;
EXPLAIN (COSTS OFF) SELECT sum(num) FROM vector_aoco_test WHERE num > 10;
EXPLAIN (COSTS OFF) SELECT sum(num) FROM vector_heap_test WHERE num > 10;
EXPLAIN (COSTS OFF) SELECT id FROM vector_aoco_test LIMIT 3;
EXPLAIN (COSTS OFF) SELECT id, id * 2 AS d FROM vector_aoco_test LIMIT 3;
SELECT sum(id * 2 + 1) FROM vector_aoco_test;
-- INT4 batch JIT: fuse arithmetic into one selection loop, preserve NULLs and
-- fall back to the ordinary evaluator on signed overflow.
SET jit = on;
SET jit_above_cost = 0;
SET jit_inline_above_cost = 0;
SET jit_optimize_above_cost = 0;
SET optimizer_jit_above_cost = 0;
SET optimizer_jit_inline_above_cost = 0;
SET optimizer_jit_optimize_above_cost = 0;
SELECT sum(id * 2 + 1) FROM vector_aoco_test;
SELECT id * 2 + 1 FROM vector_aoco_small
WHERE id IS NULL OR id BETWEEN -1000 AND 1000 ORDER BY id;
SELECT 2147483647 * 2 + 1 FROM vector_aoco_small LIMIT 1;
SELECT (-2147483648) * 2 + 1 FROM vector_aoco_small LIMIT 1;
-- Bare numeric Vars stay on the generic path; deferred arithmetic is JITted
-- with exact scale/rounding and a whole-row fallback for NaN and wide values.
SELECT sum(n), avg(n) FROM vector_numeric_test;
SELECT sum(n * 2), avg(n + 1), sum(n - 0.5), sum(n * n) FROM vector_numeric_test;
SELECT sum(n * (1 - n)), avg((1 - n) * (1 + n)) FROM vector_numeric_batch_test;
RESET jit;
RESET jit_above_cost;
RESET jit_inline_above_cost;
RESET jit_optimize_above_cost;
RESET optimizer_jit_above_cost;
RESET optimizer_jit_inline_above_cost;
RESET optimizer_jit_optimize_above_cost;
EXPLAIN (COSTS OFF) SELECT id, val FROM vector_aoco_test LIMIT 3;
SET statement_mem = '256kB';
EXPLAIN (COSTS OFF) SELECT val, count(*) FROM vector_aoco_test GROUP BY val;
RESET statement_mem;
-- ---- vectorized COPY TO: batch AOCS formatters, byte-identical output ----
CREATE TABLE vector_copy_test
(n_id numeric(12,2), n_small numeric(4,4), n_free numeric, d_val date,
 b_val bool, i2 int2, i4 int4, i8 int8, t_val text, v_val varchar(10),
 c_val bpchar(5), v_dist text)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (v_dist);
-- one INSERT per row: single-row statements insert in statement order, so
-- the AOCS row numbers (and hence the COPY TO output order) are stable
INSERT INTO vector_copy_test VALUES
('30.00', '0.0500', '123.456', '1993-06-16', true, 32767, 2147483647,
 9223372036854775807, E'a\tb', 'x', 'y', 'dist');
INSERT INTO vector_copy_test VALUES
('0.30', '0.0001', '0.00005', '1992-08-14', false, -32768, -2147483648,
 -9223372036854775808, E'c\nd', '', '', 'dist');
INSERT INTO vector_copy_test VALUES
('-0.05', '-0.9999', '99999999999999999999999999999.123456789', '0044-03-15 BC',
 NULL, 0, 0, 0, E'e\\f', 'abcdefghij', 'q', 'dist');
INSERT INTO vector_copy_test VALUES
('29034.60', '0.0000', 'NaN', 'infinity', NULL, NULL, NULL, NULL, 'g|h',
 NULL, NULL, 'dist');
INSERT INTO vector_copy_test VALUES
('-29034.60', '0.9999', NULL, '-infinity', NULL, NULL, NULL, NULL, E'i j',
 NULL, NULL, 'dist');
INSERT INTO vector_copy_test VALUES
(NULL, NULL, NULL, '5874897-12-31', NULL, NULL, NULL, NULL,
 'кириллица', NULL, NULL, 'dist');
INSERT INTO vector_copy_test VALUES
('NaN', '0.1234', '0.001', '1970-01-01', NULL, NULL, NULL, NULL,
 E'lead\ttrail ', NULL, NULL, 'dist');
INSERT INTO vector_copy_test VALUES
('0.00', '0.5000', '9999999999.99', '2000-02-29', true, 1, 2, 3, '', '',
 'mid', 'dist');
INSERT INTO vector_copy_test VALUES
('-0.0001', '0.2500', '-45.005', '0001-01-01 BC', true, 7, 7, 7,
 E'quote"q', 'lead ', 'c    ', 'dist');
-- all rows share the distribution key, so the whole table lives on one
-- segment; off and force must emit byte-identical output
SET gp_vectorized_execution = off;
\copy vector_copy_test TO STDOUT WITH (DELIMITER '|')
SET gp_vectorized_execution = force;
\copy vector_copy_test TO STDOUT WITH (DELIMITER '|')
SET gp_vectorized_execution = off;
\copy vector_copy_test (b_val, n_free, d_val, t_val, i8) TO STDOUT WITH (DELIMITER '|')
SET gp_vectorized_execution = force;
\copy vector_copy_test (b_val, n_free, d_val, t_val, i8) TO STDOUT WITH (DELIMITER '|')
-- ISO DateStyle exercises the fast date formatter
SET DateStyle = 'ISO, MDY';
SET gp_vectorized_execution = off;
\copy vector_copy_test TO STDOUT WITH (DELIMITER '|')
SET gp_vectorized_execution = force;
\copy vector_copy_test TO STDOUT WITH (DELIMITER '|')
RESET DateStyle;
-- non-ISO DateStyle must route date columns back to the output function
SET DateStyle = German;
SET gp_vectorized_execution = off;
\copy vector_copy_test TO STDOUT WITH (DELIMITER '|')
SET gp_vectorized_execution = force;
\copy vector_copy_test TO STDOUT WITH (DELIMITER '|')
RESET DateStyle;
-- heap storage stays on the regular row-by-row path
CREATE TABLE vector_copy_heap (LIKE vector_copy_test) DISTRIBUTED BY (v_dist);
INSERT INTO vector_copy_heap SELECT * FROM vector_copy_test;
SET gp_vectorized_execution = off;
\copy vector_copy_heap TO STDOUT WITH (DELIMITER '|')
SET gp_vectorized_execution = force;
\copy vector_copy_heap TO STDOUT WITH (DELIMITER '|')
RESET gp_vectorized_execution;
-- ---- vectorized COPY FROM: batch text parse, identical loaded data ----
CREATE TABLE vector_copy_from
(i2 int2, i4 int4, i8 int8, b bool, d date, n numeric(12,2), nf numeric,
 t text, vc varchar(10), v_dist text)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (v_dist);
CREATE TABLE vector_copy_from2 (LIKE vector_copy_from) DISTRIBUTED BY (v_dist);
SET gp_vectorized_execution = off;
COPY vector_copy_from FROM STDIN WITH (DELIMITER '|');
1|2|3|t|2021-02-28|1.50|123|hello|abc|dist
-32768|-2147483648|-9223372036854775808|false|2020-02-29|-0.05|+42|\tabc\|back\\slash| x y |dist
32767|2147483647|9223372036854775807|true|1999-12-31|9999999999.99|0.30|кириллица|nullfix|dist
\N|\N|\N|\N|\N|\N|\N|\N|\N|\N
7|7|7|TRUE|2021-01-01|007.00|007.000|leadzero||dist
\.
SET gp_vectorized_execution = force;
COPY vector_copy_from2 FROM STDIN WITH (DELIMITER '|');
1|2|3|t|2021-02-28|1.50|123|hello|abc|dist
-32768|-2147483648|-9223372036854775808|false|2020-02-29|-0.05|+42|\tabc\|back\\slash| x y |dist
32767|2147483647|9223372036854775807|true|1999-12-31|9999999999.99|0.30|кириллица|nullfix|dist
\N|\N|\N|\N|\N|\N|\N|\N|\N|\N
7|7|7|TRUE|2021-01-01|007.00|007.000|leadzero||dist
\.
-- both loads must produce byte-identical tables
SET gp_vectorized_execution = off;
COPY vector_copy_from TO STDOUT WITH (DELIMITER '|');
COPY vector_copy_from2 TO STDOUT WITH (DELIMITER '|');
RESET gp_vectorized_execution;
-- ---- vectorized COPY FROM with SEGMENT REJECT LIMIT ----
CREATE TABLE vector_copy_sreh (i4 int4, t text, v_dist text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (v_dist);
CREATE TABLE vector_copy_sreh2 (LIKE vector_copy_sreh) DISTRIBUTED BY (v_dist);
SET gp_vectorized_execution = off;
COPY vector_copy_sreh FROM STDIN WITH (DELIMITER '|') SEGMENT REJECT LIMIT 100 ROWS;
1|good1|dist
abc|bad-int|dist
2|good2|dist
99999999999|overflow|dist
3|good3|dist
\.
SET gp_vectorized_execution = force;
COPY vector_copy_sreh2 FROM STDIN WITH (DELIMITER '|') SEGMENT REJECT LIMIT 100 ROWS;
1|good1|dist
abc|bad-int|dist
2|good2|dist
99999999999|overflow|dist
3|good3|dist
\.
SET gp_vectorized_execution = off;
COPY vector_copy_sreh TO STDOUT WITH (DELIMITER '|');
COPY vector_copy_sreh2 TO STDOUT WITH (DELIMITER '|');
RESET gp_vectorized_execution;
-- ---- vectorized COPY TO, CSV format: quoting must be byte-identical ----
CREATE TABLE vector_copy_csv (i4 int4, d date, t text, v_dist text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (v_dist);
INSERT INTO vector_copy_csv VALUES
(1, '2021-02-28', 'plain', 'dk');
INSERT INTO vector_copy_csv VALUES
(2, '2020-02-29', 'has,comma', 'dk');
INSERT INTO vector_copy_csv VALUES
(3, '1999-12-31', E'has"quote', 'dk');
INSERT INTO vector_copy_csv VALUES
(4, '2021-01-01', E'line1\nline2', 'dk');
INSERT INTO vector_copy_csv VALUES
(5, NULL, '', 'dk');
INSERT INTO vector_copy_csv VALUES
(6, 'infinity', 'кириллица, да', 'dk');
INSERT INTO vector_copy_csv VALUES
(7, '1999-12-31', '\.', 'dk');
SET gp_vectorized_execution = off;
\copy vector_copy_csv TO STDOUT WITH (FORMAT csv)
SET gp_vectorized_execution = force;
\copy vector_copy_csv TO STDOUT WITH (FORMAT csv)
SET gp_vectorized_execution = off;
\copy vector_copy_csv TO STDOUT WITH (FORMAT csv, FORCE_QUOTE t)
SET gp_vectorized_execution = force;
\copy vector_copy_csv TO STDOUT WITH (FORMAT csv, FORCE_QUOTE t)
RESET gp_vectorized_execution;
RESET gp_vectorized_min_rows;
RESET gp_vectorized_execution;
-- ---- vectorized COPY TO: untypmod'd numeric with an exotic display scale ----
-- The fast numeric formatter renders into a fixed buffer; display scales it
-- cannot hold must take the exact output function instead of overrunning it.
CREATE TABLE vector_copy_numeric (n numeric, v_dist text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (v_dist);
-- one INSERT per row: single-row statements insert in statement order, so
-- the AOCS row numbers (and hence the COPY TO output order) are stable
INSERT INTO vector_copy_numeric VALUES
('0.1000000000000000000000001', 'dist');
INSERT INTO vector_copy_numeric VALUES
(('-2.' || repeat('9', 30))::numeric, 'dist');
INSERT INTO vector_copy_numeric VALUES
(42, 'dist');
SET gp_vectorized_execution = off;
\copy vector_copy_numeric TO STDOUT WITH (DELIMITER '|')
SET gp_vectorized_execution = force;
\copy vector_copy_numeric TO STDOUT WITH (DELIMITER '|')
RESET gp_vectorized_execution;
-- ---- vectorized COPY TO, binary format: only the regular path may write
-- the binary framing; off and force must produce byte-identical files ----
CREATE TABLE vector_copy_bin (i4 int4, t text, v_dist text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (v_dist);
INSERT INTO vector_copy_bin VALUES (1, 'x', 'dist'), (2, 'y', 'dist');
SET gp_vectorized_execution = off;
COPY vector_copy_bin TO '/tmp/vector_copy_bin.data' WITH (FORMAT binary);
SELECT encode(pg_read_binary_file('/tmp/vector_copy_bin.data'), 'hex');
SET gp_vectorized_execution = force;
COPY vector_copy_bin TO '/tmp/vector_copy_bin.data' WITH (FORMAT binary);
SELECT encode(pg_read_binary_file('/tmp/vector_copy_bin.data'), 'hex');
RESET gp_vectorized_execution;
CREATE TABLE vector_copy_bin_in (LIKE vector_copy_bin) DISTRIBUTED BY (v_dist);
COPY vector_copy_bin_in FROM '/tmp/vector_copy_bin.data' WITH (FORMAT binary);
SELECT i4, t FROM vector_copy_bin_in ORDER BY i4;
DROP TABLE vector_copy_bin_in;
\! rm -f /tmp/vector_copy_bin.data
-- ---- vectorized COPY TO, CSV under a client encoding different from the
-- database encoding: the batch formatter has no transcoder, so the regular
-- per-attribute conversion must handle these ----
CREATE TABLE vector_copy_enc (t text, v_dist text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (v_dist);
INSERT INTO vector_copy_enc VALUES ('äöü é è', 'dist'), ('plain', 'dist');
SET gp_vectorized_execution = off;
\encoding LATIN1
\copy vector_copy_enc TO STDOUT WITH (FORMAT csv)
\encoding UTF8
SET gp_vectorized_execution = force;
\encoding LATIN1
\copy vector_copy_enc TO STDOUT WITH (FORMAT csv)
\encoding UTF8
RESET gp_vectorized_execution;
DROP TABLE vector_copy_enc;
-- ---- vectorized COPY FROM: FORCE_NOT_NULL / FORCE_NULL rewrite null-marker
-- matches in the regular conversion loop, which the fast parse skips ----
CREATE TABLE vector_copy_fn (a text, b text, v_dist text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (v_dist);
CREATE TABLE vector_copy_fn2 (LIKE vector_copy_fn)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (v_dist);
SET gp_vectorized_execution = off;
COPY vector_copy_fn FROM STDIN WITH (FORMAT csv, NULL 'NULL', FORCE_NOT_NULL(a), FORCE_NULL(b));
NULL,"NULL",dist
plain,"plain",dist
\.
SET gp_vectorized_execution = force;
COPY vector_copy_fn2 FROM STDIN WITH (FORMAT csv, NULL 'NULL', FORCE_NOT_NULL(a), FORCE_NULL(b));
NULL,"NULL",dist
plain,"plain",dist
\.
-- both loads must produce byte-identical tables
SET gp_vectorized_execution = off;
COPY vector_copy_fn TO STDOUT WITH (DELIMITER '|');
COPY vector_copy_fn2 TO STDOUT WITH (DELIMITER '|');
SET gp_vectorized_execution = force;
-- ---- vectorized COPY FROM: fast numeric builder overflow falls back ----
-- A mantissa that fits the int64 accumulator can still overflow the
-- display-scale padding of the fast numeric builder; such fields keep
-- their full precision through the regular input function.
CREATE TABLE vector_copy_numovf (n numeric, v_dist text)
WITH (appendonly = true, orientation = column) DISTRIBUTED BY (v_dist);
INSERT INTO vector_copy_numovf VALUES ('0', 'dist');
COPY vector_copy_numovf FROM STDIN WITH (DELIMITER '|');
1234567890123456.7|dist
42.5|dist
\.
SET gp_vectorized_execution = off;
COPY (SELECT n, v_dist FROM vector_copy_numovf ORDER BY n) TO STDOUT WITH (DELIMITER '|');
SET gp_vectorized_execution = force;
COPY (SELECT n, v_dist FROM vector_copy_numovf ORDER BY n) TO STDOUT WITH (DELIMITER '|');
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
DROP TABLE vector_time_test;
DROP TABLE vector_numeric_test;
DROP TABLE vector_numeric_batch_test;
DROP TABLE vector_numeric_wide_test;
DROP TABLE vector_arithmetic_test;
DROP TABLE vector_rle_test;
-- ---- Text and bpchar equality ----
-- texteq/textne and bpchareq/bpcharne take a bitwise path under a
-- deterministic collation; padding, NULLs, empty strings and toasted values
-- must match the row-wise executor exactly.
CREATE TABLE vector_text_test (
    id int,
    t text,
    v varchar(20),
    b bpchar(10),
    t2 text
) WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (id);
INSERT INTO vector_text_test
SELECT i,
    CASE WHEN i % 7 = 0 THEN NULL
         WHEN i % 5 = 0 THEN ''
         WHEN i % 3 = 0 THEN 'abc  '
         ELSE 'brand#' || (i % 11) END,
    CASE WHEN i % 4 = 0 THEN NULL ELSE 'v' || (i % 13) END,
    CASE WHEN i % 6 = 0 THEN NULL
         WHEN i % 4 = 0 THEN 'x'
         ELSE 'BP' || (i % 9) END,
    CASE WHEN i % 8 = 0 THEN NULL
         ELSE 'brand#' || (i % 11) END
FROM generate_series(1, 5000) AS i;
ANALYZE vector_text_test;
SET gp_vectorized_execution = force;
-- constant comparisons: exact, padded, empty
SELECT count(*) FROM vector_text_test WHERE t = 'brand#3';
SELECT count(*) FROM vector_text_test WHERE t <> 'brand#3';
SELECT count(*) FROM vector_text_test WHERE t = '';
SELECT count(*) FROM vector_text_test WHERE t = 'abc';
SELECT count(*) FROM vector_text_test WHERE b = 'BP4';
SELECT count(*) FROM vector_text_test WHERE b <> 'BP4';
SELECT count(*) FROM vector_text_test WHERE b = 'BP4   ';
SELECT count(*) FROM vector_text_test WHERE b = 'x';
SELECT count(*) FROM vector_text_test WHERE v = 'v5';
SELECT count(*) FROM vector_text_test WHERE v <> 'v5';
SELECT count(*) FROM vector_text_test WHERE v = 'v5 ';
-- column against column, NULLs on both sides
SELECT count(*) FROM vector_text_test WHERE t = t2;
SELECT count(*) FROM vector_text_test WHERE t <> t2;
SELECT count(*) FROM vector_text_test WHERE b = b;
-- values stored out-of-line (external TOAST) through a heap table
CREATE TABLE vector_text_toast_test (id int, t text, b bpchar(10))
DISTRIBUTED BY (id);
INSERT INTO vector_text_toast_test
SELECT i,
    CASE WHEN i % 8 = 0 THEN repeat('z' || (i % 3), 1000) ELSE 'short' || i END,
    'BP' || i
FROM generate_series(1, 1000) AS i;
ANALYZE vector_text_toast_test;
SELECT count(*) FROM vector_text_toast_test WHERE t = repeat('z1', 1000);
SELECT count(*) FROM vector_text_toast_test WHERE t = 'short42';
SELECT count(*) FROM vector_text_toast_test WHERE b = 'BP100';
RESET gp_vectorized_execution;
DROP TABLE vector_text_test;
DROP TABLE vector_text_toast_test;
-- ---- Hash join on text keys ----
-- The batch probe's key equality must take the plan's hashcollations, like
-- the standard executor's hj_Collations, not the clause inputcollid: ORCA
-- leaves the latter invalid, and the vector join used to fail with "could
-- not determine which collation" where the standard executor succeeded.
CREATE TABLE vector_hj_text (k int, v text, b bpchar(10))
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (k);
INSERT INTO vector_hj_text
SELECT i % 100, 'val' || (i % 100), 'BP' || (i % 100)
FROM generate_series(1, 20000) AS i;
ANALYZE vector_hj_text;
SET gp_vectorized_execution = force;
SELECT count(*), sum(length(a.v)) FROM vector_hj_text a JOIN vector_hj_text b ON a.v = b.v;
SELECT count(*), count(a.b) FROM vector_hj_text a JOIN vector_hj_text b ON a.k = b.k AND a.v = b.v;
SELECT count(*) FROM vector_hj_text a JOIN vector_hj_text b ON a.b = b.b;
SELECT count(*) FROM vector_hj_text a LEFT JOIN vector_hj_text b ON a.v = b.v WHERE a.k < 0;
RESET gp_vectorized_execution;
DROP TABLE vector_hj_text;
-- ---- IN lists over constant sets ----
-- ScalarArrayOp with a constant array compares by bits: integer-family
-- words against a sorted unique set, text/bpchar elements unpacked once
-- (bpchar trimmed); NULL elements keep the three-valued semantics, and
-- external/compressed values fall back to the per-element function calls.
CREATE TABLE vector_in_set (i2 int2, i4 int4, i8 int8, b bool, d date,
                            t text, v varchar(20), c bpchar(10), tbig text)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (i4);
INSERT INTO vector_in_set
SELECT (i % 50)::int2, i % 100, (i % 70)::int8, (i % 2 = 0),
       date '2020-01-01' + (i % 40),
       CASE WHEN i % 7 = 0 THEN NULL WHEN i % 5 = 0 THEN '' ELSE 't' || (i % 9) END,
       CASE WHEN i % 4 = 0 THEN NULL ELSE 'v' || (i % 11) END,
       CASE WHEN i % 6 = 0 THEN NULL ELSE 'BP' || (i % 8) END,
       CASE WHEN i % 8 = 0 THEN repeat('x', 3000) ELSE 'big' || (i % 6) END
FROM generate_series(1, 9000) AS i;
ANALYZE vector_in_set;
SET gp_vectorized_execution = force;
SELECT count(*) FROM vector_in_set WHERE i4 IN (1,2,3,99,100,50,49,48,47,46,45,44,43,42,41,40,39,38);
SELECT count(*) FROM vector_in_set WHERE i4 NOT IN (1,2,3);
SELECT count(*) FROM vector_in_set WHERE i2 IN (1::int2,2::int2,49::int2);
SELECT count(*) FROM vector_in_set WHERE i8 IN (1::int8,69::int8);
SELECT count(*) FROM vector_in_set WHERE b IN (true,false);
SELECT count(*) FROM vector_in_set WHERE d IN (date '2020-01-02', date '2020-02-05');
SELECT count(*) FROM vector_in_set WHERE t IN ('t1','t2');
SELECT count(*) FROM vector_in_set WHERE t NOT IN ('t1','t2');
SELECT count(*) FROM vector_in_set WHERE v IN ('v1',NULL,'v2');
SELECT count(*) FROM vector_in_set WHERE v NOT IN ('v1',NULL,'v2');
SELECT count(*) FROM vector_in_set WHERE c IN ('BP1','BP7   ');
SELECT count(*) FROM vector_in_set WHERE c NOT IN ('BP1');
SELECT count(*) FROM vector_in_set WHERE tbig IN ('big1','big5');
SELECT count(*) FROM vector_in_set WHERE tbig IN (repeat('x',3000));
RESET gp_vectorized_execution;
DROP TABLE vector_in_set;
-- ---- LIKE over constant patterns ----
-- Patterns of literals and % compile into ordered pieces matched by
-- memchr/memcmp; patterns with _ or an escape character, dynamic
-- patterns, and nondeterministic collations stay on the generic matcher.
CREATE TABLE vector_like (t text, c bpchar(12), v text)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (t);
INSERT INTO vector_like
SELECT CASE WHEN i % 11 = 0 THEN NULL
            WHEN i % 7 = 0 THEN ''
            WHEN i % 5 = 0 THEN 'express requests are carefully express'
            WHEN i % 3 = 0 THEN 'Customer#XComplaints'
            ELSE 'ord' || (i % 13) || 'x' || (i % 17) || 'final' END,
       CASE WHEN i % 4 = 0 THEN NULL ELSE 'BC' || (i % 6) END,
       CASE WHEN i % 8 = 0 THEN repeat('y', 2500) ELSE 'va' || (i % 9) || 'l' END
FROM generate_series(1, 9000) AS i;
ANALYZE vector_like;
SET gp_vectorized_execution = force;
SELECT count(*) FROM vector_like WHERE t LIKE '%express%';
SELECT count(*) FROM vector_like WHERE t NOT LIKE '%express%';
SELECT count(*) FROM vector_like WHERE t LIKE 'ord%';
SELECT count(*) FROM vector_like WHERE t LIKE '%final';
SELECT count(*) FROM vector_like WHERE t LIKE 'ord%x%final';
SELECT count(*) FROM vector_like WHERE t LIKE '%';
SELECT count(*) FROM vector_like WHERE t LIKE '';
SELECT count(*) FROM vector_like WHERE t LIKE 'ord1x2final';
-- literal patterns without % are equality tests: a proper prefix of an
-- existing value must not match
SELECT count(*) FROM vector_like WHERE t LIKE 'ord1x2fina';
SELECT count(*) FROM vector_like WHERE t NOT LIKE 'ord1x2fina';
SELECT count(*) FROM vector_like WHERE t LIKE 'ord';
SELECT count(*) FROM vector_like WHERE v LIKE 'va3';
SELECT count(*) FROM vector_like WHERE t LIKE '%Customer%Complaints%';
SELECT count(*) FROM vector_like WHERE t LIKE '%express%express';
SELECT count(*) FROM vector_like WHERE t LIKE 'exp_ress%';
SELECT count(*) FROM vector_like WHERE t LIKE 'exp\%ress%';
SELECT count(*) FROM vector_like WHERE c LIKE 'BC1%';
SELECT count(*) FROM vector_like WHERE c NOT LIKE 'BC1%';
SELECT count(*) FROM vector_like WHERE v LIKE 'va3%';
SELECT count(*) FROM vector_like WHERE v LIKE '%y%';
SELECT count(*) FROM vector_like WHERE t IS NOT NULL AND t LIKE '%' || 'express' || '%';
RESET gp_vectorized_execution;
DROP TABLE vector_like;
-- ---- Nested Loop prefetch rescan over a scan-root region ----
-- A motion below the join sets prefetch_inner: the NL fetches one inner row
-- under a fake outer and rescans immediately.  A scan-root vector region
-- must rewind its scan on rescan, not drop the rest of the current batch.
CREATE TABLE vector_nl_prefetch (a int, v numeric(12,2))
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (a);
INSERT INTO vector_nl_prefetch
SELECT i, (i % 1000)::numeric(12,2) FROM generate_series(1, 50000) AS i;
ANALYZE vector_nl_prefetch;
SET gp_vectorized_execution = force;
EXPLAIN (COSTS OFF)
SELECT (a % 7) AS g, count(*) FROM vector_nl_prefetch
WHERE (a % 5) <> 0 AND v > (SELECT avg(v) FROM vector_nl_prefetch)
GROUP BY g ORDER BY g;
SELECT (a % 7) AS g, count(*) FROM vector_nl_prefetch
WHERE (a % 5) <> 0 AND v > (SELECT avg(v) FROM vector_nl_prefetch)
GROUP BY g ORDER BY g;
RESET gp_vectorized_execution;
DROP TABLE vector_nl_prefetch;
-- ---- Three-valued logic in vector AND/OR quals ----
-- A row whose non-final conjunct is NULL must not pass an AND because the
-- final argument is true (NULL AND true is NULL, not true); likewise a NULL
-- disjunct must stay NULL when the remaining arguments are all false.
CREATE TABLE vector_boolnull (a int, b char(10))
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (a);
INSERT INTO vector_boolnull
SELECT i, CASE WHEN i % 10 = 0 THEN NULL ELSE 'Women' END
FROM generate_series(1, 20000) AS i;
INSERT INTO vector_boolnull VALUES (30000, NULL), (30001, NULL);
ANALYZE vector_boolnull;
SET gp_vectorized_execution = force;
EXPLAIN (COSTS OFF)
SELECT count(*) FROM vector_boolnull
WHERE b = 'Women' AND (a = 30 OR a = 31);
SELECT count(*) FROM vector_boolnull
WHERE b = 'Women' AND (a = 30 OR a = 31);
SELECT count(*) FROM vector_boolnull
WHERE b = 'Women' AND a = 30000;
SELECT count(*) FROM vector_boolnull
WHERE NOT (b = 'Women' OR a = 30001);
SELECT count(*) FROM vector_boolnull
WHERE NOT (b = 'Nope' OR a = 30001);
RESET gp_vectorized_execution;
DROP TABLE vector_boolnull;
-- ---- Correlated subplan results must outlive the per-row evaluation ----
-- A pass-by-reference subplan result (e.g. a numeric average) lives in the
-- subplan's per-tuple memory, which the next row's execution resets.  The
-- vector qual evaluates a whole batch before any row is consumed, so the
-- result must be copied to durable storage; otherwise every row compares
-- against the last row's value.
CREATE TABLE vector_subplan_lifetime (a int, cat char(10), p numeric(7,2))
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (a);
INSERT INTO vector_subplan_lifetime
SELECT i, ('C' || (i % 7))::char(10), (10 + (i % 47))::numeric(7,2)
FROM generate_series(1, 20000) AS i;
ANALYZE vector_subplan_lifetime;
SET gp_vectorized_execution = force;
EXPLAIN (COSTS OFF)
SELECT count(*) FROM vector_subplan_lifetime t
WHERE t.p > 1.2 * (SELECT avg(u.p) FROM vector_subplan_lifetime u
                   WHERE u.cat = t.cat);
SELECT count(*) FROM vector_subplan_lifetime t
WHERE t.p > 1.2 * (SELECT avg(u.p) FROM vector_subplan_lifetime u
                   WHERE u.cat = t.cat);
SELECT t.cat, count(*), sum(t.p) FROM vector_subplan_lifetime t
WHERE t.p > 1.2 * (SELECT avg(u.p) FROM vector_subplan_lifetime u
                   WHERE u.cat = t.cat)
GROUP BY t.cat ORDER BY t.cat;
RESET gp_vectorized_execution;
DROP TABLE vector_subplan_lifetime;
-- ---- Nested Loop prefetch rescan over an aggregate region ----
-- Same prefetch-rewind case as vector_nl_prefetch above, but the inner
-- region root is a blocking aggregate: the rescan must reset the aggregate
-- state along with the leaf scan, or the join sees an empty inner side.
CREATE TABLE vector_nl_prefetch_agg (a int, g int, v numeric(12,2))
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (a);
INSERT INTO vector_nl_prefetch_agg
SELECT i, i % 11, (i % 1000)::numeric(12,2) FROM generate_series(1, 60000) AS i;
ANALYZE vector_nl_prefetch_agg;
SET gp_vectorized_execution = force;
SET enable_hashjoin = off;
SET enable_mergejoin = off;
SET optimizer_enable_hashjoin = off;
SET optimizer_enable_mergejoin = off;
EXPLAIN (COSTS OFF)
SELECT count(*) FROM
  (SELECT avg(v) AS m FROM vector_nl_prefetch_agg WHERE g = 1 GROUP BY g) o,
  (SELECT g, avg(v) AS s FROM vector_nl_prefetch_agg GROUP BY g) i
WHERE i.s > 0.9 * o.m;
SELECT count(*) FROM
  (SELECT avg(v) AS m FROM vector_nl_prefetch_agg WHERE g = 1 GROUP BY g) o,
  (SELECT g, avg(v) AS s FROM vector_nl_prefetch_agg GROUP BY g) i
WHERE i.s > 0.9 * o.m;
RESET optimizer_enable_hashjoin;
RESET optimizer_enable_mergejoin;
RESET enable_hashjoin;
RESET enable_mergejoin;
RESET gp_vectorized_execution;
DROP TABLE vector_nl_prefetch_agg;
-- ---- Fixed-length pass-by-reference columns over batch motion ----
-- timetz, interval and name are fixed-length but pass-by-reference; the
-- batch motion codec must measure them by typlen, not VARSIZE_ANY, or the
-- sender dereferences the value bytes as a varlena header and crashes.
CREATE TABLE vector_fixedref (a int, t timetz, n name, i interval)
WITH (appendonly = true, orientation = column)
DISTRIBUTED BY (a);
INSERT INTO vector_fixedref
SELECT i, ('01:0' || (i % 10) || ':00+0' || (i % 3))::timetz,
       ('n' || i)::name, (i || ' seconds')::interval
FROM generate_series(1, 1000) AS i;
ANALYZE vector_fixedref;
SET gp_vectorized_execution = force;
SELECT count(*), count(t), min(t), max(t) FROM vector_fixedref;
SELECT n, i FROM vector_fixedref WHERE a = 7;
RESET gp_vectorized_execution;
DROP TABLE vector_fixedref;
DROP TABLE vector_float_test;
DROP TABLE vector_append_parent CASCADE;
DROP TABLE vector_part_test CASCADE;
DROP TABLE vector_drop_test CASCADE;
DROP TABLE vector_union_a, vector_union_b, vector_union_h, vector_union_hj CASCADE;
DROP TABLE vector_copy_test;
DROP TABLE vector_copy_heap;
DROP TABLE vector_copy_from;
DROP TABLE vector_copy_from2;
DROP TABLE vector_copy_sreh;
DROP TABLE vector_copy_sreh2;
DROP TABLE vector_copy_csv;
DROP TABLE vector_copy_numeric;
DROP TABLE vector_copy_bin;
DROP TABLE vector_copy_fn;
DROP TABLE vector_copy_fn2;
DROP TABLE vector_copy_numovf;
DROP TABLE vector_f4sum;
-- ---- A join-rooted region must not leak its probe scan to the motion sender ----
-- The motion batch sender pulls a region's output batches directly, and a
-- NestLoop-rooted region's output is the joined rows, not its probe scan's
-- batches.  The batchable-output gates declined only hash joins, so the
-- sender handed the probe scan's raw batches to the gather and the query
-- lost every row, while count(*) over the same join stayed correct through
-- the row path.  The projection must stay in place with no ORDER BY: either
-- would put a row-wise node between the region and the motion and mask the
-- bug.  The tables stay unanalyzed on purpose: with default estimates the
-- planner broadcasts the inner side, keeping the probe a local scan, which
-- is the shape the region wrap needs (analyzed sizes make it broadcast the
-- probe side instead, and the row-mode join hides the bug).  Every output
-- row renders identically, so no ordering is needed for a deterministic
-- diff.
CREATE TABLE vector_nl_pull_a (f1 boolean) DISTRIBUTED BY (f1);
CREATE TABLE vector_nl_pull_b (f1 boolean) DISTRIBUTED BY (f1);
INSERT INTO vector_nl_pull_a SELECT true FROM generate_series(1, 3);
INSERT INTO vector_nl_pull_b SELECT false FROM generate_series(1, 4);
SET gp_vectorized_execution = force;
EXPLAIN (COSTS OFF)
SELECT a.f1, b.f1 FROM vector_nl_pull_a a, vector_nl_pull_b b
WHERE b.f1 <> a.f1;
SELECT a.f1, b.f1 FROM vector_nl_pull_a a, vector_nl_pull_b b
WHERE b.f1 <> a.f1;
RESET gp_vectorized_execution;
DROP TABLE vector_nl_pull_a;
DROP TABLE vector_nl_pull_b;
-- ---- The COPY fast parse must reject rows the regular path rejects ----
-- With the fast parse engaged (any gp_vectorized_execution other than
-- off), two dispatch-split holes silently accepted bad rows: a line with
-- more fields than the table dropped the extra ones, and a short line
-- whose missing column the QE still owed (a middle distribution key
-- keeps the QD prefix shorter than the line) loaded with NULLs.  Both
-- must error exactly as they do with the feature off, and clean loads
-- must keep working.  The rows come in over COPY FROM STDIN, inline.
CREATE TABLE vector_cpf_1 (c1 text) DISTRIBUTED BY (c1);
CREATE TABLE vector_cpf_2 (c1 text, c2 text) DISTRIBUTED BY (c2);
CREATE TABLE vector_cpf_3m (c1 text, c2 text, c3 text) DISTRIBUTED BY (c2);
SET gp_vectorized_execution = force;
COPY vector_cpf_1 FROM STDIN WITH (DELIMITER ',');
1,text1
\.
COPY vector_cpf_3m FROM STDIN WITH (DELIMITER ',');
1,text1
\.
COPY vector_cpf_2 FROM STDIN WITH (DELIMITER ',');
1,text1
\.
SELECT count(*) FROM vector_cpf_1;
SELECT count(*) FROM vector_cpf_2;
SELECT c1, c2 FROM vector_cpf_2;
SELECT count(*) FROM vector_cpf_3m;
RESET gp_vectorized_execution;
DROP TABLE vector_cpf_1;
DROP TABLE vector_cpf_2;
DROP TABLE vector_cpf_3m;
-- ---- Whole-row vars and composite constructors ----
-- A whole-row var reaches the row reader as attno 0; the deform set
-- computed from pulled var numbers marked no column at all, so every
-- whole-row expression assembled an all-zero tuple (and a ConvertRowtype
-- over it crashed in the standard rowtype conversion).  The view wraps
-- row(whole-row)::composite over a typed table, the exact create_view
-- shape that failed.
CREATE TYPE vector_wholerow_rec AS (q1 int8, q2 int8);
CREATE TABLE vector_wholerow_t OF vector_wholerow_rec;
INSERT INTO vector_wholerow_t VALUES (123, 456), (123, 4567890123456789);
CREATE TYPE vector_wholerow_nested AS (x vector_wholerow_rec);
CREATE VIEW vector_wholerow_v AS
  SELECT row(i)::vector_wholerow_nested FROM vector_wholerow_t i;
SET gp_vectorized_execution = force;
SELECT i FROM vector_wholerow_t i;
SELECT row(i) FROM vector_wholerow_t i;
SELECT * FROM vector_wholerow_v;
RESET gp_vectorized_execution;
DROP VIEW vector_wholerow_v;
DROP TABLE vector_wholerow_t;
DROP TYPE vector_wholerow_nested;
DROP TYPE vector_wholerow_rec;
-- ---- Grouping sets in the vector aggregate ----
-- Three shapes were wrong: a mixed aggregate with two or more plain
-- (non-hash) sets crashed - the plain-set walk assumed one phase per
-- set and ran off the phases array; a plain aggregate with grouping
-- sets emitted only the first set's row; and with several plain sets
-- every input row advanced only the first set's state, so the sets
-- collapsed into one row with multiplied aggregates (this is the
-- groupingsets test's gstest_empty shape that crashed a segment).
CREATE TABLE vector_gsets_empty (a int, b int, v numeric) DISTRIBUTED BY (a);
CREATE TABLE vector_gsets_rows (a int, b int, v numeric) DISTRIBUTED BY (a);
INSERT INTO vector_gsets_rows VALUES (1, 2, 3.5), (1, 3, 4.5);
SET gp_vectorized_execution = force;
SELECT a, b, sum(v), count(*) FROM vector_gsets_empty
GROUP BY GROUPING SETS ((a, b), (), (), ())
ORDER BY 1 NULLS FIRST, 2 NULLS FIRST;
SELECT a, b, sum(v), count(*) FROM vector_gsets_rows
GROUP BY GROUPING SETS ((a, b), (), (), ())
ORDER BY 1 NULLS FIRST, 2 NULLS FIRST;
SELECT a, b, sum(v), count(*) FROM vector_gsets_rows
GROUP BY GROUPING SETS ((a), (), (a, b), ())
ORDER BY 1 NULLS FIRST, 2 NULLS FIRST;
SELECT sum(v), count(*) FROM vector_gsets_rows
GROUP BY GROUPING SETS ((), (), ());
SELECT sum(v), count(*) FROM vector_gsets_rows;
RESET gp_vectorized_execution;
DROP TABLE vector_gsets_empty;
DROP TABLE vector_gsets_rows;
-- ---- Array coercion keeps the source array's shape ----
-- The vector interpreter rebuilt a coerced array with one dimension and
-- lower bound 1, dropping the source's ndim, dims and lower bound.
-- int2vector -> int2[] carries lower bound 0, so values coerced inside a
-- vector region compared unequal to the same coercion done on the row
-- path (this is what made gpcheckcat report phantom pg_index
-- inconsistencies); 2-D arrays came back flattened.
CREATE TEMPORARY TABLE vector_arrcoerce_c AS
  SELECT gp_segment_id AS segid, indexrelid, indkey FROM pg_index
  DISTRIBUTED BY (segid);
SET gp_vectorized_execution = force;
SELECT indexrelid, v::text AS v, array_lower(v, 1) AS lb, count(*)
FROM (
  SELECT segid, indexrelid, indkey::int2[] AS v
    FROM vector_arrcoerce_c WHERE indexrelid = 112
  UNION ALL
  SELECT gp_segment_id, indexrelid, indkey::int2[] AS v
    FROM gp_dist_random('pg_index') WHERE indexrelid = 112) u
GROUP BY 1, 2, 3 ORDER BY 1, 2, 3;
-- Empty int2vectors - the distribution keys of partitioned children -
-- must keep their shape too: a region path that rebuilt the empty as a
-- zero-dimensional array split identical gp_distribution_policy rows
-- into different groups (phantom gpcheckcat inconsistencies).
CREATE TABLE vector_arrcoerce_part (id int) DISTRIBUTED RANDOMLY
  PARTITION BY RANGE (id) (START (0) END (20) EVERY (10));
CREATE TEMPORARY TABLE vector_arrcoerce_d AS
  SELECT p.gp_segment_id AS segid, p.localoid, p.distkey
  FROM gp_distribution_policy p
  JOIN pg_class c ON c.oid = p.localoid
  WHERE c.relname LIKE 'vector_arrcoerce_part%' DISTRIBUTED BY (segid);
SELECT v::text AS txt, array_lower(v, 1) AS lb, array_ndims(v) AS nd
FROM (
  SELECT segid, distkey::int2[] AS v FROM vector_arrcoerce_d
  UNION ALL
  SELECT gp_segment_id, distkey::int2[] AS v FROM gp_dist_random('gp_distribution_policy')) u
GROUP BY 1, 2, 3 ORDER BY 1, 2, 3;
RESET gp_vectorized_execution;
DROP TABLE vector_arrcoerce_c;
DROP TABLE vector_arrcoerce_d;
DROP TABLE vector_arrcoerce_part;
-- ---- Nested join chain: the probe feed survives the child chain's EOF ----
-- A hash join wrapped over another hash join region pulls its probe
-- batches through VectorRegionProbeSourceFeed.  At the child chain's EOF
-- the pull returns true with a NULL batch; the feed must treat that as
-- its own EOF.  Pre-fix it dereferenced the NULL and crashed a segment
-- on the first query whose plan probed a region through the feed.
CREATE TABLE vector_chain_f (k1 int, k2 int, v numeric) DISTRIBUTED BY (k1);
CREATE TABLE vector_chain_d1 (k int, a text) DISTRIBUTED BY (k);
CREATE TABLE vector_chain_d2 (k int, b text) DISTRIBUTED BY (k);
INSERT INTO vector_chain_f
  SELECT i, i % 10 + 1, i * 0.5 FROM generate_series(1, 3000) i;
INSERT INTO vector_chain_f VALUES (40000, 77, 1.0);
INSERT INTO vector_chain_d1 SELECT i, 'd1-' || i FROM generate_series(1, 50) i;
INSERT INTO vector_chain_d2 SELECT i, 'd2-' || i FROM generate_series(1, 30) i;
SET gp_vectorized_execution = force;
SELECT count(*) FROM vector_chain_f f
  JOIN vector_chain_d1 d1 ON f.k1 = d1.k
  JOIN vector_chain_d2 d2 ON f.k2 = d2.k;
-- an inner chain that yields zero rows hits the feed's EOF on the very
-- first pull, where the NULL deref crashed before any batch was copied
SELECT count(*) FROM vector_chain_f f
  JOIN vector_chain_d1 d1 ON f.k1 = d1.k
  JOIN vector_chain_d2 d2 ON f.k2 = d2.k
  WHERE d1.a = 'no such key';
RESET gp_vectorized_execution;
DROP TABLE vector_chain_f;
DROP TABLE vector_chain_d1;
DROP TABLE vector_chain_d2;
