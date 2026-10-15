// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: 'https://ammarsalahi.github.io/rust-ai-examples',
	integrations: [
		starlight({
			title: 'Rust AI Examples',
			defaultLocale: 'root',
			locales: {
				root: { label: 'English', lang: 'en' },
				fa: { label: 'فارسی', lang: 'fa', dir: 'rtl' },
			},
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/ammarsalahi/rust-ai-examples' }],
			sidebar: [
				{
					label: 'Introduction',
					translations: { fa: 'معرفی' },
					items: [
						{
							label: 'Getting Started',
							translations: { fa: 'شروع به کار' },
							slug: 'introduction/getting-started',
						},
						{
							label: 'Project Overview',
							translations: { fa: 'نمای کلی پروژه' },
							slug: 'introduction/overview',
						},
					],
				},
				{
					label: 'Machine Learning',
					translations: { fa: 'یادگیری ماشین' },
					items: [
						{
							label: 'Decision Tree Classifier',
							translations: { fa: 'طبقه‌بند درخت تصمیم' },
							slug: 'machine-learning/decision-tree-classifier',
						},
						{
							label: 'House Price Prediction',
							translations: { fa: 'پیش‌بینی قیمت خانه' },
							slug: 'machine-learning/house-price-prediction',
						},
						{
							label: 'KNN Classifier',
							translations: { fa: 'طبقه‌بند KNN' },
							slug: 'machine-learning/knn-classifier',
						},
					],
				},
				{
					label: 'Deep Learning',
					translations: { fa: 'یادگیری عمیق' },
					items: [
						{
							label: 'Image Classification',
							translations: { fa: 'طبقه‌بندی تصویر' },
							slug: 'deep-learning/image-classification',
						},
						{
							label: 'OCR',
							translations: { fa: 'تشخیص نویسه نوری (OCR)' },
							slug: 'deep-learning/ocr',
						},
					],
				},
				{
					label: 'NLP',
					translations: { fa: 'پردازش زبان طبیعی' },
					items: [
						{
							label: 'Keyword Extraction',
							translations: { fa: 'استخراج کلیدواژه' },
							slug: 'nlp/keyword-extraction',
						},
						{
							label: 'Language Detection',
							translations: { fa: 'تشخیص زبان' },
							slug: 'nlp/language-detection',
						},
						{
							label: 'Sentiment Analysis',
							translations: { fa: 'تحلیل احساسات' },
							slug: 'nlp/sentiment-analysis',
						},
						{
							label: 'Text Classification',
							translations: { fa: 'طبقه‌بندی متن' },
							slug: 'nlp/text-classification',
						},
						{
							label: 'Email Spam Classification',
							translations: { fa: 'طبقه‌بندی هرزنامهٔ ایمیل' },
							slug: 'nlp/email-spam-classification',
						},
					],
				},
				{
					label: 'Reference',
					translations: { fa: 'مرجع' },
					items: [{ autogenerate: { directory: 'reference' } }],
				},
			],
		}),
	],
});
