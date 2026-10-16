package com.quickpitik.architecture

import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PhotoRemovalHistoryRepository
import com.tngtech.archunit.base.DescribedPredicate
import com.tngtech.archunit.core.domain.JavaClasses
import com.tngtech.archunit.core.domain.JavaMethodCall
import com.tngtech.archunit.core.importer.ClassFileImporter
import com.tngtech.archunit.core.importer.ImportOption
import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * The boundary rules from the 2026-09-11 architecture audit that a static check
 * can express *without false positives* (plan §4 P2.1). The audit's other
 * findings — a transaction held across provider I/O, `@Value` outside `config/`
 * — are not statically decidable and stay review items, deliberately not
 * automated here.
 *
 * Each exemption below is a named class with the reason it is allowed; a new
 * violator is a build failure, not a new exemption by reflex.
 */
class BoundariesTest {

    private val production: JavaClasses = ClassFileImporter()
        .withImportOption(ImportOption.Predefined.DO_NOT_INCLUDE_TESTS)
        .importPackages("com.quickpitik")

    /**
     * Every rule below is a `noClasses(...)`, which an empty import satisfies
     * trivially. This is the self-check that the importer actually found the
     * production classes — without it a packaging change could turn the whole
     * file green and silent.
     */
    @Test
    fun `the importer loaded the production classes and excluded the test classes`() {
        assertTrue(production.any { it.name == "com.quickpitik.controller.AuthController" })
        assertTrue(production.none { it.simpleName.endsWith("Test") })
    }

    @Test
    fun `controllers never depend on repositories`() {
        noClasses().that().resideInAPackage("com.quickpitik.controller..")
            .should().dependOnClassesThat().resideInAPackage("com.quickpitik.repository..")
            .because(
                "a controller binds, validates, gates by role, rate-limits and makes one service call; " +
                    "data access belongs to the owning service (backend/AGENTS.md boundary 1)",
            )
            .check(production)
    }

    @Test
    fun `only the earnings domain depends on TransactionRepository`() {
        noClasses().that().resideOutsideOfPackage("com.quickpitik.service.earnings..")
            // Read-only exemptions: both reach the ledger through sum/count/
            // aggregate projections for admin screens and call no writer.
            // Cross-domain reads for hydration are allowed; writes are not.
            .and().doNotHaveSimpleName("AdminSalesService")
            .and().doNotHaveSimpleName("AdminPayoutService")
            .should().dependOnClassesThat().haveSimpleName("TransactionRepository")
            .because(
                "`transactions` has one writing service — service/earnings/TransactionMintingService " +
                    "(backend/AGENTS.md boundary 3, audit P-1)",
            )
            .check(production)
    }

    @Test
    fun `only the earnings domain depends on PayoutStatusHistoryRepository`() {
        noClasses().that().resideOutsideOfPackage("com.quickpitik.service.earnings..")
            .should().dependOnClassesThat().haveSimpleName("PayoutStatusHistoryRepository")
            .because("the payout audit trail is written only by service/earnings/PayoutStatusRecorder")
            .check(production)
    }

    /**
     * `payout_cycles` is read for hydration by five services across three
     * domains, so a dependency rule here would be five exemptions of legitimate
     * reads. The invariant that actually matters is the *write*, so this rule
     * matches the call, not the type.
     */
    @Test
    fun `only the earnings domain writes payout cycles`() {
        noClasses().that().resideOutsideOfPackage("com.quickpitik.service.earnings..")
            .should().callMethodWhere(WRITES_PAYOUT_CYCLES)
            .because(
                "payout_cycles moves only through service/earnings/PayoutCycleTransitions (37f214f5, audit P-2)",
            )
            .check(production)
    }

    @Test
    fun `only the removal history recorder appends removal history`() {
        assertTrue(production.any { it.simpleName == "PhotoRemovalHistoryRecorder" })
        noClasses().that().doNotHaveFullyQualifiedName("com.quickpitik.service.removal.PhotoRemovalHistoryRecorder")
            .should().callMethodWhere(WRITES_REMOVAL_HISTORY)
            .because("photo removal history is append-only and has one writing service")
            .check(production)
    }

    @Test
    fun `removal history exposes no application update delete or save method`() {
        assertTrue(PhotoRemovalHistoryRepository::class.java.methods.none {
            it.name.startsWith("save") || it.name.startsWith("delete") || it.name.startsWith("update")
        })
    }

    private companion object {
        val WRITES_REMOVAL_HISTORY =
            object : DescribedPredicate<JavaMethodCall>("an append call on PhotoRemovalHistoryRepository") {
                override fun test(call: JavaMethodCall): Boolean =
                    call.targetOwner.isAssignableTo(PhotoRemovalHistoryRepository::class.java) &&
                        call.target.name == "append"
            }
        /**
         * The AdminPayoutService methods that still save `payout_cycles` directly,
         * none of them a status move: `generateForWeek` creates cycles and refreshes
         * a PENDING row's amount; `saveFirstApproval` stamps the first approver
         * on a cycle that stays PENDING; `markProcessing` claims the provider before
         * the transfer leaves. Matched by method (lambdas compile to `name$lambda$n`)
         * so a new method in that class that saves directly still fails the build.
         */
        val DIRECT_CYCLE_SAVES = setOf("generateForWeek", "saveFirstApproval", "markProcessing")

        val WRITES_PAYOUT_CYCLES =
            object : DescribedPredicate<JavaMethodCall>("a write call on PayoutCycleRepository") {
                override fun test(call: JavaMethodCall): Boolean =
                    call.targetOwner.isAssignableTo(PayoutCycleRepository::class.java) &&
                        (call.target.name.startsWith("save") || call.target.name.startsWith("delete") || call.target.name == "insert") &&
                        !(call.originOwner.simpleName == "AdminPayoutService" &&
                            call.origin.name.substringBefore('$') in DIRECT_CYCLE_SAVES)
            }
    }
}
