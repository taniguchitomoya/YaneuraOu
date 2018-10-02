// NNUE評価関数の層Dropoutの定義

// C++やNNUEやディープラーニングについて深い理解のないままコピペで書いたので
// 間違いを含む可能性があるのはもちろんのこと、コピペ元由来の必要のないコードや足りてない部分もあるかもしれない


#ifndef _NNUE_LAYERS_DROPOUT_H_
#define _NNUE_LAYERS_DROPOUT_H_

#include "../../../shogi.h"

#if defined(EVAL_NNUE)

#include "../nnue_common.h"

namespace Eval {

	namespace NNUE {

		namespace Layers {

			// Dropout
			template <typename PreviousLayer>
			class Dropout {
			public:
				// 入出力の型
				using InputType = typename PreviousLayer::OutputType;
				using OutputType = InputType;
				//static_assert(std::is_same<InputType, std::int32_t>::value, "");

				// 入出力の次元数
				static constexpr IndexType kInputDimensions =
					PreviousLayer::kOutputDimensions;
				static constexpr IndexType kOutputDimensions = kInputDimensions;

				// この層で使用する順伝播用バッファのサイズ
				static constexpr std::size_t kSelfBufferSize =
					CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);

				// 入力層からこの層までで使用する順伝播用バッファのサイズ
				static constexpr std::size_t kBufferSize =
					PreviousLayer::kBufferSize + kSelfBufferSize;

				// 評価関数ファイルに埋め込むハッシュ値
				static constexpr std::uint32_t GetHashValue() {
					//評価関数ファイルをtnk_wcsc28互換とするため、あえてhash値を変更しない
					return PreviousLayer::GetHashValue();

					/*
					std::uint32_t hash_value = 0x0DDD1234u;
					hash_value += PreviousLayer::GetHashValue();
					return hash_value;
					*/
				}

				// 入力層からこの層までの構造を表す文字列
				static std::string GetStructureString() {
					//評価関数ファイルをtnk_wcsc28互換とするため、あえて文字列を変更しない
					return PreviousLayer::GetStructureString();
					/*
					return "Dropout[" +
						std::to_string(kOutputDimensions) + "](" +
						PreviousLayer::GetStructureString() + ")";
					*/
				}

				// パラメータを読み込む
				bool ReadParameters(std::istream& stream) {
					return previous_layer_.ReadParameters(stream);
				}

				// パラメータを書き込む
				bool WriteParameters(std::ostream& stream) const {
					return previous_layer_.WriteParameters(stream);
				}

				// 順伝播
				const OutputType* Propagate(
					const TransformedFeatureType* transformed_features, char* buffer) const {
					const auto input = previous_layer_.Propagate(
						transformed_features, buffer + kSelfBufferSize);
					const auto output = reinterpret_cast<OutputType*>(buffer);

					for (IndexType i = 0; i < kOutputDimensions; ++i) {
						output[i] = input[i] / 2; //output[i] * ( 1 -  dropout ratio(0.5) )
					}
					return output;
				}

			private:
				// 学習用クラスをfriendにする
				friend class Trainer<Dropout>;

				// この層の直前の層
				PreviousLayer previous_layer_;
			};

		}  // namespace Layers

	}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)

#endif
