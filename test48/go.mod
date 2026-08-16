module github.com/openfluke/loom/arcagitesting/test48

go 1.22.5

require github.com/openfluke/welvet v0.0.0

require github.com/openfluke/webgpu v1.0.4 // indirect

replace github.com/openfluke/welvet => ../../../../../../chaosglue/welvet

replace github.com/openfluke/webgpu => ../../../../webgpu

replace github.com/eliben/go-sentencepiece => ../../../../../../chaosglue/welvet/third_party/go-sentencepiece
