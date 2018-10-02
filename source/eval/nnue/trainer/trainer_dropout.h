// NNUE評価関数の学習クラステンプレートのDropout用特殊化

// C++やNNUEやディープラーニングについて深い理解のないままコピペで書いたので
// 間違いを含む可能性があるのはもちろんのこと、コピペ元由来の必要のないコードや足りてない部分もあるかもしれない

#ifndef _NNUE_TRAINER_DROPOUT_H_
#define _NNUE_TRAINER_DROPOUT_H_

#include "../../../shogi.h"

#if defined(EVAL_LEARN) && defined(EVAL_NNUE)

#include "../../../learn/learn.h"
#include "../layers/dropout.h"
#include "trainer.h"

#include <random>

namespace Eval {

	namespace NNUE {

		template <typename PreviousLayer>
		class Trainer<Layers::Dropout<PreviousLayer>> {
		private:
			// 学習対象の層の型
			using LayerType = Layers::Dropout<PreviousLayer>;
			//random.ここ単独で持つのではなく、他の場所と共用した方が良いかもしれない
			std::mt19937 mtrandom;

			//dropoutのフラグ保存用
			bool dropout_mask[LayerType::kOutputDimensions];


		public:
			// ファクトリ関数
			static std::shared_ptr<Trainer> Create(
				LayerType* target_layer, FeatureTransformer* feature_transformer) {
				return std::shared_ptr<Trainer>(
					new Trainer(target_layer, feature_transformer));
			}

			// ハイパーパラメータなどのオプションを設定する
			void SendMessage(Message* message) {
				previous_layer_trainer_->SendMessage(message);
			}

			// パラメータを乱数で初期化する
			template <typename RNG>
			void Initialize(RNG& rng) {
				previous_layer_trainer_->Initialize(rng);
			}

			// 順伝播
			const LearnFloatType* Propagate(const std::vector<Example>& batch) {
				if (output_.size() < kOutputDimensions * batch.size()) {
					output_.resize(kOutputDimensions * batch.size());
					gradients_.resize(kInputDimensions * batch.size());
				}
				const auto input = previous_layer_trainer_->Propagate(batch);

				std::uniform_real_distribution<double> zeroone(0.0, 1.0);

				for (IndexType i = 0; i < kOutputDimensions; ++i) {
					dropout_mask[i] = zeroone(mtrandom) < 0.5;
				}

				batch_size_ = static_cast<IndexType>(batch.size());
				for (IndexType b = 0; b < batch_size_; ++b) {
					const IndexType batch_offset = kOutputDimensions * b;
					for (IndexType i = 0; i < kOutputDimensions; ++i) {
						const IndexType index = batch_offset + i;
						output_[index] = input[index] * dropout_mask[i];
					}
				}
				return output_.data();
			}

			// 逆伝播
			void Backpropagate(const LearnFloatType* gradients,
				LearnFloatType learning_rate) {
				for (IndexType b = 0; b < batch_size_; ++b) {
					const IndexType batch_offset = kOutputDimensions * b;
					for (IndexType i = 0; i < kOutputDimensions; ++i) {
						const IndexType index = batch_offset + i;
						gradients_[index] = gradients[index] * dropout_mask[i];
					}
				}
				previous_layer_trainer_->Backpropagate(gradients_.data(), learning_rate);
			}

		private:
			// コンストラクタ
			Trainer(LayerType* target_layer, FeatureTransformer* feature_transformer) :
				batch_size_(0),
				previous_layer_trainer_(Trainer<PreviousLayer>::Create(
					&target_layer->previous_layer_, feature_transformer)),
				target_layer_(target_layer) {
			}

			// 入出力の次元数
			static constexpr IndexType kInputDimensions = LayerType::kOutputDimensions;
			static constexpr IndexType kOutputDimensions = LayerType::kOutputDimensions;


			// ミニバッチのサンプル数
			IndexType batch_size_;

			// 直前の層のTrainer
			const std::shared_ptr<Trainer<PreviousLayer>> previous_layer_trainer_;

			// 学習対象の層
			LayerType* const target_layer_;

			// 順伝播用バッファ
			std::vector<LearnFloatType> output_;

			// 逆伝播用バッファ
			std::vector<LearnFloatType> gradients_;

		};

	}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_LEARN) && defined(EVAL_NNUE)

#endif
