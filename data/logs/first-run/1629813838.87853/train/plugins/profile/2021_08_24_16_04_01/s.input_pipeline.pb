  *G???Ԭ?@D?l?[ ?@2?
]Iterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch::BatchV2::Shuffle::ParallelMapV2??ut @!d??TB@)??ut @1d??TB@:Preprocessing2?
lIterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch::BatchV2::Shuffle::ParallelMapV2::ParallelMapV2ni5$?1??!rpA?7?@@)ni5$?1??1rpA?7?@@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch::BatchV2::Shuffle::ParallelMapV2::ParallelMapV2::FlatMap[2]::TFRecord??&3?V??!??|?0@)??&3?V??1??|?0@:Advanced file read2|
EIterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch::BatchV2??????@!n?'?c?H@)?҇.?o??1h???b(@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch::BatchV2::Shuffle???߃ @!z*???B@)j???ک?1??c?i??:Preprocessing2?
uIterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch::BatchV2::Shuffle::ParallelMapV2::ParallelMapV2::FlatMap?4?\???!Yx???0@)?r.?U??1ڗ??`E??:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteSkipu?ɧ??!????#???)??????1-j?"?H??:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch????S??!M??^i???)????S??1M??^i???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?#????!???qND??)?????g?1J2?U٪?:Preprocessing2F
Iterator::Model'?_[???!???ڢ??)?/???"e?1b4?|9??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.