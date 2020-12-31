#include "gtest/gtest.h"

#include "../src/extraction/Snippet.h"

TEST(SnippetTestSuite, TestInitOK)
{
    Snippet<short, 1, 1> snippet;
}

TEST(SnippetTestSuite, TestGet)
{
    Snippet<short, 1, 1> snippet;
    EXPECT_EQ(0, snippet.get(0, 0));
}
