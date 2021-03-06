#include "gtest/gtest.h"
#include "test_utilities/test_utilities.cuh"

#include "../src/extraction/snippet_extractor.cuh"

/*
 *
 */
TEST(SnippetExtractorTest, InitialState) {
  Params params;
  auto probe = probe_from_env();

  SnippetExtractor<short> extractor(params, probe);
  extractor.ExtractSnippets();

  EXPECT_EQ(0, extractor.frame_offset());
}