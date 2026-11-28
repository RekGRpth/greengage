-- Robustness of vectorized regions under interruption and early shutdown:
-- a statement timeout and a user cancel must interrupt a long-running
-- vectorized join region and leave the session usable, and closing a cursor
-- above a vector region must shut it down cleanly.

1:CREATE TABLE vec_iso (id int4, grp int4) WITH (appendonly = true, orientation = column) DISTRIBUTED BY (id);
1:INSERT INTO vec_iso SELECT g, g % 10 FROM generate_series(1, 20000) AS g;
1:ANALYZE vec_iso;
1:SET gp_vectorized_execution = force;

-- A statement timeout interrupts the running vectorized join region.
1:SET statement_timeout = '2s';
1:SELECT count(*) FROM vec_iso a CROSS JOIN vec_iso b WHERE a.id + b.id >= 0;
1:RESET statement_timeout;

-- The session stays usable after the interruption.
1:SELECT count(*) FROM vec_iso WHERE id < 3;

-- A user cancel interrupts the running vectorized join region.
1&:SELECT count(*) FROM vec_iso a CROSS JOIN vec_iso b WHERE a.id + b.id >= 0;
2:SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE query = 'SELECT count(*) FROM vec_iso a CROSS JOIN vec_iso b WHERE a.id + b.id >= 0;';
1<:

-- The session stays usable after the cancellation, too.
1:SELECT count(*) FROM vec_iso WHERE id < 5;

-- Closing a cursor above a vector region squelches it early.
1:BEGIN;
1:DECLARE veccur CURSOR FOR SELECT id, grp FROM vec_iso WHERE id % 7 = 0 ORDER BY id LIMIT 200;
1:FETCH 3 FROM veccur;
1:CLOSE veccur;
1:END;

-- The region still answers a plain query afterwards.
1:SELECT count(*) FROM vec_iso WHERE id % 2 = 0 AND id <= 100;

-- An injected fault in the batch scan fails the query cleanly.
2:SELECT gp_inject_fault('aocs_getnext_batch', 'error', dbid) FROM gp_segment_configuration WHERE role = 'p' AND content = 0;
1:SELECT count(*) FROM vec_iso WHERE id % 2 = 0;
2:SELECT gp_inject_fault('aocs_getnext_batch', 'reset', dbid) FROM gp_segment_configuration WHERE role = 'p' AND content = 0;

-- The same query succeeds after the reset.
1:SELECT count(*) FROM vec_iso WHERE id % 2 = 0;

-- The fault leaves the row executor untouched.
2:SELECT gp_inject_fault('aocs_getnext_batch', 'error', dbid) FROM gp_segment_configuration WHERE role = 'p' AND content = 0;
1:SET gp_vectorized_execution = off;
1:SELECT count(*) FROM vec_iso WHERE id % 2 = 0;
1:SET gp_vectorized_execution = force;
2:SELECT gp_inject_fault('aocs_getnext_batch', 'reset', dbid) FROM gp_segment_configuration WHERE role = 'p' AND content = 0;

-- An injected fault in the vectorized spill path fails the query cleanly:
-- under ORCA the aggregate runs inside the region (the Postgres planner
-- leaves it above the region, on the standard executor), so this section
-- pins the optimizer; under a small statement_mem the vectorized hash
-- aggregate spills groups and the armed fault fires on the first write.
1:SET optimizer = on;
1:SET statement_mem = '512kB';
2:SELECT gp_inject_fault('vector_agg_spill', 'error', dbid) FROM gp_segment_configuration WHERE role = 'p' AND content = 0;
1:SELECT count(*) FROM (SELECT id % 9999 AS k, count(*) FROM vec_iso GROUP BY 1) t;
2:SELECT gp_inject_fault('vector_agg_spill', 'reset', dbid) FROM gp_segment_configuration WHERE role = 'p' AND content = 0;

-- The same spilling query succeeds after the reset.
1:SELECT count(*) FROM (SELECT id % 9999 AS k, count(*) FROM vec_iso GROUP BY 1) t;
1:RESET statement_mem;

-- The armed fault leaves the standard executor's spill path untouched.
2:SELECT gp_inject_fault('vector_agg_spill', 'error', dbid) FROM gp_segment_configuration WHERE role = 'p' AND content = 0;
1:SET gp_vectorized_execution = off;
1:SET statement_mem = '512kB';
1:SELECT count(*) FROM (SELECT id % 9999 AS k, count(*) FROM vec_iso GROUP BY 1) t;
1:RESET statement_mem;
1:SET gp_vectorized_execution = force;
2:SELECT gp_inject_fault('vector_agg_spill', 'reset', dbid) FROM gp_segment_configuration WHERE role = 'p' AND content = 0;
1:RESET optimizer;

1:DROP TABLE vec_iso;
