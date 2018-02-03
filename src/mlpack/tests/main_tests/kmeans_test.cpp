/**
 * @file kmeans_test.cpp
 * @author Prabhat Sharma
 *
 * Test mlpackMain() of kmeans_main.cpp
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "Kmeans";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/kmeans/kmeans_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct KmTestFixture
{
public:
    KmTestFixture()
    {
        // Cache in the options for this program.
        CLI::RestoreSettings(testName);
    }

    ~KmTestFixture()
    {
        // Clear the settings.
        CLI::ClearSettings();
    }
};

void ResetSettings()
{
    CLI::ClearSettings();
    CLI::RestoreSettings(testName);
}

BOOST_FIXTURE_TEST_SUITE(KmeansMainTest, KmTestFixture);


/**
 * Checking that all the algorithms yield same results
 */
    BOOST_AUTO_TEST_CASE(AlgorithmsSimilarTest)
    {
        constexpr int N = 100;
        constexpr int D = 4;
        constexpr int C=5;
        std::string algo="naive";
        arma::mat InputData = arma::randu<arma::mat>(N, D);
        arma::mat OutputData;
        arma::mat Centroid;

        SetInputParam("input_file", std::move(InputData));
        SetInputParam("output_file",std::move(OutputData));
        SetInputParam("centroid_file",std::move(Centroid));
        SetInputParam("clusters", std::move(C));
        SetInputParam("algorithm", std::move(algo));

        mlpackMain();

        const arma::mat NaiveOutput = CLI::GetParam<arma::mat>("output");

        ResetSettings();

        algo="elkan";

        SetInputParam("input_file", std::move(InputData));
        SetInputParam("output_file",std::move(OutputData));
        SetInputParam("centroid_file",std::move(Centroid));
        SetInputParam("clusters", std::move(C));
        SetInputParam("algorithm", std::move(algo));

        mlpackMain();

        const arma::mat ElkanOutput = CLI::GetParam<arma::mat>("output");

        ResetSettings();

        algo="hamerly";

        SetInputParam("input_file", std::move(InputData));
        SetInputParam("output_file",std::move(OutputData));
        SetInputParam("centroid_file",std::move(Centroid));
        SetInputParam("clusters", std::move(C));
        SetInputParam("algorithm", std::move(algo));

        mlpackMain();

        const arma::mat HamerlyOutput = CLI::GetParam<arma::mat>("output");

        ResetSettings();

        algo="dualtree";

        SetInputParam("input_file", std::move(InputData));
        SetInputParam("output_file",std::move(OutputData));
        SetInputParam("centroid_file",std::move(Centroid));
        SetInputParam("clusters", std::move(C));
        SetInputParam("algorithm", std::move(algo));

        mlpackMain();

        const arma::mat DualTreeOutput = CLI::GetParam<arma::mat>("output");

        ResetSettings();

        algo="dualtree-covertree";

        SetInputParam("input_file", std::move(InputData));
        SetInputParam("output_file",std::move(OutputData));
        SetInputParam("centroid_file",std::move(Centroid));
        SetInputParam("clusters", std::move(C));
        SetInputParam("algorithm", std::move(algo));

        mlpackMain();

        const arma::mat DualCoverTreeOutput = CLI::GetParam<arma::mat>("output");

        //Check That all the algorithms yield the same clusters
        CheckMatrices(NaiveOutput,ElkanOutput);
        CheckMatrices(ElkanOutput,HamerlyOutput);
        CheckMatrices(HamerlyOutput,DualTreeOutput);
        CheckMatrices(DualTreeOutput,DualCoverTreeOutput);
    }


/**
 * Checking that number of Clusters are non negative
 */
    BOOST_AUTO_TEST_CASE(NonNegativeClustersTest)
    {
        constexpr int N = 10;
        constexpr int D = 4;

        arma::mat InputData = arma::randu<arma::mat>(N, D);
        arma::mat OutputData;
        arma::mat Centroid;
        SetInputParam("input_file", std::move(InputData));
        SetInputParam("output_file",std::move(OutputData));
        SetInputParam("centroid_file",std::move(Centroid));
        SetInputParam("clusters", (int) -1); //Invalid

        Log::Fatal.ignoreInput = true;
        BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
        Log::Fatal.ignoreInput = false;
    }

/**
 * Checking that number of Clusters is Integer
 */
    BOOST_AUTO_TEST_CASE(IntegerClustersTest)
    {
        constexpr int N = 10;
        constexpr int D = 4;

        arma::mat InputData = arma::randu<arma::mat>(N, D);
        arma::mat OutputData;
        arma::mat Centroid;
        SetInputParam("input_file", std::move(InputData));
        SetInputParam("output_file",std::move(OutputData));
        SetInputParam("centroid_file",std::move(Centroid));
        SetInputParam("clusters", (double) 0.5); //Invalid

        Log::Fatal.ignoreInput = true;
        BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
        Log::Fatal.ignoreInput = false;
    }

/**
 * Checking that number of Maximum Iterations is Integer
 */
    BOOST_AUTO_TEST_CASE(MaximumIterationIntegerTest)
    {
        constexpr int N = 10;
        constexpr int D = 4;

        arma::mat InputData = arma::randu<arma::mat>(N, D);
        arma::mat OutputData;
        arma::mat Centroid;
        SetInputParam("input_file", std::move(InputData));
        SetInputParam("output_file",std::move(OutputData));
        SetInputParam("centroid_file",std::move(Centroid));
        SetInputParam("max_iterations", (double) 1000.50); //Invalid

        Log::Fatal.ignoreInput = true;
        BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
        Log::Fatal.ignoreInput = false;
    }

/**
 * Checking that number of percentage is between 0 and 1 when --refined_start is specified
 */
    BOOST_AUTO_TEST_CASE(RefinedStartPercentageTest)
    {
        constexpr int N = 10;
        constexpr int D = 4;

        arma::mat InputData = arma::randu<arma::mat>(N, D);
        arma::mat OutputData;
        arma::mat Centroid;
        SetInputParam("input_file", std::move(InputData));
        SetInputParam("output_file",std::move(OutputData));
        SetInputParam("centroid_file",std::move(Centroid));
        SetInputParam("refined_start", true);
        SetInputParam("clusters", (double) 2.0); //Invalid

        Log::Fatal.ignoreInput = true;
        BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
        Log::Fatal.ignoreInput = false;
    }

/**
 * Checking that that size and dimensionality of prediction is correct.
 */
    BOOST_AUTO_TEST_CASE(KmClusteringSizeCheck)
    {
        constexpr int N = 10;
        constexpr int D = 4;
        constexpr int C = 2;

        arma::mat InputData = arma::randu<arma::mat>(N, D);
        arma::mat OutputData;
        arma::mat Centroid;
        SetInputParam("input_file", std::move(InputData));
        SetInputParam("output_file",std::move(OutputData));
        SetInputParam("centroid_file",std::move(Centroid));
        SetInputParam("clusters",std::move(C));


        mlpackMain();

        const arma::mat output = CLI::GetParam<arma::mat>("output");
        const arma::mat centroid = CLI::GetParam<arma::mat>("centroid");

        BOOST_REQUIRE_EQUAL(output.n_rows, N);
        BOOST_REQUIRE_EQUAL(output.n_cols, D+1);
        BOOST_REQUIRE_EQUAL(centroid.n_rows, C);
        BOOST_REQUIRE_EQUAL(centroid.n_cols,D);
    }


/**
 * Checking that that size and dimensionality of Final Input File is correct when flag --in_place is specified
 */
    BOOST_AUTO_TEST_CASE(KmClusteringResultSizeCheck)
    {
        constexpr int N = 10;
        constexpr int D = 4;
        constexpr int C = 2;

        arma::mat InputData = arma::randu<arma::mat>(N, D);
        arma::mat OutputData;
        arma::mat Centroid;
        SetInputParam("input_file", std::move(InputData));
        SetInputParam("output_file",std::move(OutputData));
        SetInputParam("centroid_file",std::move(Centroid));
        SetInputParam("clusters",std::move(C));
        SetInputParam("in_place", true);


        mlpackMain();

        BOOST_REQUIRE_EQUAL(InputData.n_cols,D+1);
        BOOST_REQUIRE_EQUAL(InputData.n_rows,N);
    }


/**
 * Ensuring that absence of Input is checked.
 */
    BOOST_AUTO_TEST_CASE(KmNoInputData)
    {
        constexpr int N = 10;
        constexpr int D = 4;
        constexpr int C=2;
        arma::mat input = arma::randu<arma::mat>(N,D);
        arma::mat output;
        arma::mat centroid;

        SetInputParam("output_file",std::move(output));
        SetInputParam("centroid_file",std::move(centroid));
        SetInputParam("clusters",std::move(C));

        Log::Fatal.ignoreInput = true;
        BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
        Log::Fatal.ignoreInput = false;
    }

/**
 * Ensuring that absence of Number of Clusters is checked.
 */
    BOOST_AUTO_TEST_CASE(KmClustersNotDefined)
    {
        constexpr int N = 10;
        constexpr int D = 4;
        arma::mat input = arma::randu<arma::mat>(N,D);
        arma::mat output;
        arma::mat centroid;
        SetInputParam("input_file", std::move(input));
        SetInputParam("output_file",std::move(output));
        SetInputParam("centroid_file",std::move(centroid));

        Log::Fatal.ignoreInput = true;
        BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
        Log::Fatal.ignoreInput = false;
    }

BOOST_AUTO_TEST_SUITE_END();


