/*-------------------------------------------------------------------------
 *
 * vectorize.c
 *	  Replace eligible segment-local plan regions with vectorized regions.
 *
 * The pass runs after either optimizer has produced a complete PlannedStmt.
 * Operator-specific discovery is introduced together with the corresponding
 * batch executor.  Until then both non-off modes leave the plan unchanged.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "cdb/cdbvars.h"
#include "optimizer/planner.h"
#include "utils/guc.h"

PlannedStmt *
VectorizePlan(PlannedStmt *stmt)
{
	if (gp_vectorized_execution == GP_VECTORIZED_EXECUTION_OFF ||
		Gp_role != GP_ROLE_DISPATCH ||
		stmt == NULL || stmt->commandType != CMD_SELECT ||
		stmt->planTree == NULL)
		return stmt;

	/* No operator is eligible until the AOCS batch scan is connected. */
	return stmt;
}
