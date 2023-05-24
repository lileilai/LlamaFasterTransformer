file(REMOVE_RECURSE
  "unittest"
  "unittest.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/unittest.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
