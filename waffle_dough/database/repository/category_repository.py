from waffle_dough.database.model import Category
from waffle_dough.database.repository.base_repository import CRUDBase
from waffle_dough.field import CategoryInfo, UpdateCategoryInfo


class CategoryRepository(CRUDBase[Category, CategoryInfo, UpdateCategoryInfo]):
    pass


category_repository = CategoryRepository(Category)
